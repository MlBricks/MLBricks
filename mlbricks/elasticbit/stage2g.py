# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""ElasticBit Stage-2G Universal Core.

IMPORTANT: The algorithm below is extracted from
ElasticBit_Reduce_Benchmark_Stage2G_v2.ipynb.

No rewrite rule has been generalized for the package.  The only production
adaptation is removing notebook-only readiness prints and exposing the exact
notebook reduce wrapper through ElasticBit.reduce().
"""

import torch
from torch import nn
import torch.nn.functional as F

STAGE2G_NOTEBOOK_SOURCE = "ElasticBit_Reduce_Benchmark_Stage2G_v2.ipynb"
STAGE2G_NOTEBOOK_CELL_SHA256 = {
    "core": "61c4b679c03adcea1e1d0fac46431c96827c6dc651a7b6923340f2fc49c974cd",
    "transactional_hotfix": "bb70cd67ede7e5914df7e771bad7fa5300058c9c33a4f3b933e8b9e1da5c8287",
    "public_reduce": "b321f3c4ebe3da0eb9bf9fd72b6dd983464f3c24b3a60489767e4751085b74a6",
}


# ============================================================
# 22. UNIVERSAL STAGE-2 POLICY / FX TRACER
# ============================================================

import operator
import copy
from dataclasses import dataclass
from typing import Any

import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

_STAGE2_BARRIER_MODULES = [
    nn.ReLU,
    nn.SiLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.LayerNorm,
]
if hasattr(nn, "RMSNorm"):
    _STAGE2_BARRIER_MODULES.append(nn.RMSNorm)

STAGE2_BARRIER_MODULES = tuple(_STAGE2_BARRIER_MODULES)

STAGE2_BARRIER_FUNCTIONS = {
    F.relu,
    torch.relu,
    F.silu,
    F.gelu,
    torch.sigmoid,
    torch.tanh,
    F.softmax,
    F.layer_norm,
}

class Stage2Tracer(fx.Tracer):
    """
    Conservative tracer:
    - standard Linear layers are leaves;
    - known nonlinear boundaries are leaves;
    - ElasticBit-like matrix-backed modules are leaves so the analyzer
      can identify them without tracing through their runtime internals.
    """
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, (nn.Linear,) + STAGE2_BARRIER_MODULES):
            return True
        if hasattr(m, "matrix"):
            return True
        return super().is_leaf_module(m, module_qualified_name)

def stage2_symbolic_trace(model):
    root = copy.deepcopy(model).eval()
    tracer = Stage2Tracer()
    graph = tracer.trace(root)
    return fx.GraphModule(root, graph)

def s2_get_module(gm, node):
    if node.op != "call_module":
        return None
    return gm.get_submodule(str(node.target))

def s2_is_linear(gm, node):
    return node.op == "call_module" and isinstance(s2_get_module(gm, node), nn.Linear)

def s2_is_matrix_backed(gm, node):
    if node.op != "call_module":
        return False
    m = s2_get_module(gm, node)
    return m is not None and not isinstance(m, nn.Linear) and hasattr(m, "matrix")

def s2_is_linear_like(gm, node):
    return s2_is_linear(gm, node) or s2_is_matrix_backed(gm, node)

def s2_is_barrier(gm, node):
    if node.op == "call_module":
        return isinstance(s2_get_module(gm, node), STAGE2_BARRIER_MODULES)
    if node.op == "call_function":
        return node.target in STAGE2_BARRIER_FUNCTIONS
    return False

def s2_last_dim(node):
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return None
    return int(tm.shape[-1])

def s2_unique_name(gm, base):
    name = base
    i = 0
    while hasattr(gm, name):
        i += 1
        name = f"{base}_{i}"
    return name

def s2_flat_tensors(x):
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(s2_flat_tensors(v))
        return out
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out.extend(s2_flat_tensors(v))
        return out
    return []

def s2_compare_outputs(a, b, rtol=2e-3, atol=2e-3):
    aa = s2_flat_tensors(a)
    bb = s2_flat_tensors(b)
    if len(aa) != len(bb):
        return float("inf"), False

    rel = []
    ok = True

    for x, y in zip(aa, bb):
        d = x.float() - y.float()
        rel.append(
            float(
                torch.linalg.vector_norm(d)
                / (torch.linalg.vector_norm(x.float()) + 1e-12)
            )
        )
        ok = ok and torch.allclose(x, y, rtol=rtol, atol=atol)

    return (max(rel) if rel else 0.0), ok




# ============================================================
# 23. RIGHT-TO-LEFT AFFINE CANONICALIZER
# ============================================================

@dataclass
class S2Expr:
    op: str
    args: tuple = ()
    W: Any = None
    b: Any = None
    value: Any = None
    name: str = ""

def S2Input(name="x"):
    return S2Expr("input", name=name)

def S2Linear(x, W, b=None, name="linear"):
    return S2Expr("linear", (x,), W=W, b=b, name=name)

def S2Add(a, b):
    return S2Expr("add", (a, b))

def S2Bias(x, b):
    return S2Expr("bias", (x,), b=b)

def S2Scale(x, scale):
    return S2Expr("scale", (x,), value=float(scale))

def s2_bias_vector(b, out_dim, device, dtype):
    if b is None:
        return torch.zeros(out_dim, device=device, dtype=dtype)

    if torch.is_tensor(b):
        b = b.detach().to(device=device, dtype=dtype)
        if b.ndim == 0:
            return b.expand(out_dim)
        if b.numel() != out_dim:
            raise ValueError(
                f"Bias size {b.numel()} does not match output size {out_dim}"
            )
        return b.reshape(out_dim)

    return torch.full(
        (out_dim,),
        float(b),
        device=device,
        dtype=dtype,
    )

def s2_reduce_affine(expr, input_dim, device, compose_dtype=torch.float32):
    """
    Convert an affine dependency tree into one W,b:
        expr(x) == W @ x + b

    The walk never crosses a boundary because FX->S2Expr translation
    turns a boundary into the source signal of the next region.
    """
    if expr.op == "input":
        W = torch.eye(
            input_dim,
            device=device,
            dtype=compose_dtype,
        )
        b = torch.zeros(
            input_dim,
            device=device,
            dtype=compose_dtype,
        )
        return W, b

    if expr.op == "linear":
        W0, b0 = s2_reduce_affine(
            expr.args[0],
            input_dim,
            device,
            compose_dtype,
        )

        W = expr.W.detach().to(
            device=device,
            dtype=compose_dtype,
        )
        b = s2_bias_vector(
            expr.b,
            W.shape[0],
            device,
            compose_dtype,
        )

        return W @ W0, W @ b0 + b

    if expr.op == "add":
        W1, b1 = s2_reduce_affine(
            expr.args[0],
            input_dim,
            device,
            compose_dtype,
        )
        W2, b2 = s2_reduce_affine(
            expr.args[1],
            input_dim,
            device,
            compose_dtype,
        )

        if W1.shape != W2.shape or b1.shape != b2.shape:
            raise ValueError(
                f"Add shape mismatch: {tuple(W1.shape)} vs {tuple(W2.shape)}"
            )

        return W1 + W2, b1 + b2

    if expr.op == "bias":
        W, b = s2_reduce_affine(
            expr.args[0],
            input_dim,
            device,
            compose_dtype,
        )
        return W, b + s2_bias_vector(
            expr.b,
            b.numel(),
            device,
            compose_dtype,
        )

    if expr.op == "scale":
        W, b = s2_reduce_affine(
            expr.args[0],
            input_dim,
            device,
            compose_dtype,
        )
        s = float(expr.value)
        return s * W, s * b

    raise ValueError(f"Unsupported affine op: {expr.op}")

def s2_fx_to_expr(
    gm,
    node,
    sources,
    touched_linears,
    touched_nodes,
):
    """
    Reverse-walk an FX graph.

    Unsupported/dynamic/nonlinear nodes become boundary sources.
    This is deliberately conservative.
    """
    if not isinstance(node, fx.Node):
        raise TypeError(f"Expected FX Node, got {type(node)}")

    touched_nodes.add(node)

    # A previous nonlinearity/dynamic boundary becomes the input signal
    # of the current affine region.
    if node.op == "placeholder" or s2_is_barrier(gm, node):
        sources.add(node)
        return S2Input(node.name)

    if s2_is_linear(gm, node):
        touched_linears.add(node)
        m = s2_get_module(gm, node)

        return S2Linear(
            s2_fx_to_expr(
                gm,
                node.args[0],
                sources,
                touched_linears,
                touched_nodes,
            ),
            m.weight,
            m.bias,
            node.name,
        )

    if node.op == "call_function" and node.target in {
        operator.add,
        torch.add,
    }:
        a, b = node.args[:2]

        if isinstance(a, fx.Node) and isinstance(b, (int, float)):
            return S2Bias(
                s2_fx_to_expr(
                    gm, a, sources, touched_linears, touched_nodes
                ),
                b,
            )

        if isinstance(b, fx.Node) and isinstance(a, (int, float)):
            return S2Bias(
                s2_fx_to_expr(
                    gm, b, sources, touched_linears, touched_nodes
                ),
                a,
            )

        if isinstance(a, fx.Node) and isinstance(b, fx.Node):
            return S2Add(
                s2_fx_to_expr(
                    gm, a, sources, touched_linears, touched_nodes
                ),
                s2_fx_to_expr(
                    gm, b, sources, touched_linears, touched_nodes
                ),
            )

    if node.op == "call_function" and node.target in {
        operator.mul,
        torch.mul,
    }:
        a, b = node.args[:2]

        if isinstance(a, fx.Node) and isinstance(b, (int, float)):
            return S2Scale(
                s2_fx_to_expr(
                    gm, a, sources, touched_linears, touched_nodes
                ),
                b,
            )

        if isinstance(b, fx.Node) and isinstance(a, (int, float)):
            return S2Scale(
                s2_fx_to_expr(
                    gm, b, sources, touched_linears, touched_nodes
                ),
                a,
            )

    # Anything not proven affine becomes a hard local boundary.
    sources.add(node)
    return S2Input(node.name)

class Stage2PackedLinear(nn.Module):
    """
    A normal dense reference implementation of a packed/resultant
    projection. ElasticBit will eventually need a segmented packed
    kernel instead of this dense reference module.
    """
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)




# ============================================================
# 24. EXACT REWRITE RULES + HARD GUARDS
# ============================================================

def s2_boundary_candidate(
    gm,
    boundary,
    example_args,
    max_param_growth=0.0,
    min_linears=2,
    max_resultant_bytes=512 * 1024**2,
):
    ShapeProp(gm).propagate(*example_args)

    sources = set()
    linears = set()
    nodes = set()

    expr = s2_fx_to_expr(
        gm,
        boundary.args[0],
        sources,
        linears,
        nodes,
    )

    if len(sources) != 1:
        return None, {
            "accepted": False,
            "kind": "boundary",
            "reason": f"{len(sources)} boundary sources",
        }

    if len(linears) < min_linears:
        return None, {
            "accepted": False,
            "kind": "boundary",
            "reason": "too few linear maps",
        }

    source = next(iter(sources))

    # Consumer-safety rule:
    # every region node we plan to remove may only feed this region
    # or the final boundary.
    region_eliminate = set(nodes) - set(sources)

    for node in region_eliminate:
        external_users = [
            user
            for user in node.users
            if user not in region_eliminate
            and user is not boundary
        ]

        if external_users:
            return None, {
                "accepted": False,
                "kind": "boundary",
                "reason": "external consumer safety",
                "node": node.name,
            }

    input_dim = s2_last_dim(source)
    output_dim = s2_last_dim(boundary.args[0])

    if input_dim is None or output_dim is None:
        return None, {
            "accepted": False,
            "kind": "boundary",
            "reason": "unknown source/output shape",
        }

    modules = [s2_get_module(gm, node) for node in linears]
    first = modules[0]

    old_params = sum(
        sum(p.numel() for p in module.parameters())
        for module in modules
    )

    # Cost is checked BEFORE the potentially expensive matrix products.
    new_params = input_dim * output_dim + output_dim

    if new_params > old_params * (1.0 + max_param_growth):
        return None, {
            "accepted": False,
            "kind": "boundary",
            "reason": "cost guard",
            "old_params": int(old_params),
            "new_params": int(new_params),
        }

    new_bytes = new_params * first.weight.element_size()

    if new_bytes > max_resultant_bytes:
        return None, {
            "accepted": False,
            "kind": "boundary",
            "reason": "materialization byte guard",
            "new_bytes": int(new_bytes),
        }

    W, b = s2_reduce_affine(
        expr,
        input_dim=input_dim,
        device=first.weight.device,
        compose_dtype=torch.float32,
    )

    W = W.to(first.weight.dtype)
    b = b.to(first.weight.dtype)

    return (
        source,
        W,
        b,
        linears,
        nodes,
    ), {
        "accepted": True,
        "kind": "boundary",
        "old_params": int(old_params),
        "new_params": int(new_params),
        "linears": int(len(linears)),
    }

def s2_apply_boundary(
    gm,
    boundary,
    example_args,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    original = copy.deepcopy(gm)

    candidate, info = s2_boundary_candidate(
        gm,
        boundary,
        example_args,
        max_param_growth=max_param_growth,
        max_resultant_bytes=max_resultant_bytes,
    )

    if candidate is None:
        return gm, info

    source, W, b, _, _ = candidate

    fused = nn.Linear(
        W.shape[1],
        W.shape[0],
        bias=True,
        device=W.device,
        dtype=W.dtype,
    )

    with torch.no_grad():
        fused.weight.copy_(W)
        fused.bias.copy_(b)

    fused.requires_grad_(False)

    name = s2_unique_name(gm, "_stage2_boundary")
    gm.add_submodule(name, fused)

    graph = gm.graph

    with graph.inserting_before(boundary):
        fused_node = graph.call_module(
            name,
            args=(source,),
        )

    boundary_args = list(boundary.args)
    boundary_args[0] = fused_node
    boundary.args = tuple(boundary_args)

    graph.eliminate_dead_code()
    gm.recompile()

    if hasattr(gm, "delete_all_unused_submodules"):
        gm.delete_all_unused_submodules()

    with torch.inference_mode():
        y_original = original(*example_args)
        y_reduced = gm(*example_args)

    rel_l2, verified = s2_compare_outputs(
        y_original,
        y_reduced,
    )

    info["relative_l2"] = float(rel_l2)

    if not verified:
        info["accepted"] = False
        info["reason"] = "verification failed"
        return original, info

    return gm, info


def s2_find_same_input_fanouts(gm, linear_like=False):
    candidates = []

    for node in gm.graph.nodes:
        if linear_like:
            users = [
                user
                for user in node.users
                if s2_is_linear_like(gm, user)
            ]
        else:
            users = [
                user
                for user in node.users
                if s2_is_linear(gm, user)
            ]

        if len(users) >= 2:
            candidates.append((node, users))

    return candidates


def s2_apply_same_input_pack(
    gm,
    source,
    branches,
    example_args,
    max_packed_bytes=512 * 1024**2,
):
    original = copy.deepcopy(gm)
    modules = [s2_get_module(gm, node) for node in branches]

    if (
        len(branches) < 2
        or not all(isinstance(m, nn.Linear) for m in modules)
    ):
        return gm, {
            "accepted": False,
            "kind": "same_input_pack",
            "reason": "not enough compatible nn.Linear branches",
        }

    if len({m.in_features for m in modules}) != 1:
        return gm, {
            "accepted": False,
            "kind": "same_input_pack",
            "reason": "input feature mismatch",
        }

    packed_params = sum(
        m.weight.numel()
        + (0 if m.bias is None else m.bias.numel())
        for m in modules
    )

    element_size = modules[0].weight.element_size()

    if packed_params * element_size > max_packed_bytes:
        return gm, {
            "accepted": False,
            "kind": "same_input_pack",
            "reason": "packed byte guard",
        }

    W = torch.cat(
        [m.weight.detach() for m in modules],
        dim=0,
    )

    if any(m.bias is not None for m in modules):
        bias_parts = []

        for m in modules:
            if m.bias is None:
                bias_parts.append(
                    torch.zeros(
                        m.out_features,
                        device=m.weight.device,
                        dtype=m.weight.dtype,
                    )
                )
            else:
                bias_parts.append(m.bias.detach())

        b = torch.cat(bias_parts, dim=0)
    else:
        b = None

    split_sizes = [m.out_features for m in modules]

    name = s2_unique_name(gm, "_stage2_pack")
    gm.add_submodule(
        name,
        Stage2PackedLinear(W, b),
    )

    graph = gm.graph

    with graph.inserting_after(source):
        packed = graph.call_module(
            name,
            args=(source,),
        )

    with graph.inserting_after(packed):
        split = graph.call_function(
            torch.split,
            args=(packed, split_sizes),
            kwargs={"dim": -1},
        )

    previous = split
    replacements = []

    for index in range(len(branches)):
        with graph.inserting_after(previous):
            item = graph.call_function(
                operator.getitem,
                args=(split, index),
            )

        previous = item
        replacements.append(item)

    for old_node, new_node in zip(branches, replacements):
        old_node.replace_all_uses_with(new_node)

    for node in branches:
        graph.erase_node(node)

    graph.eliminate_dead_code()
    gm.recompile()

    if hasattr(gm, "delete_all_unused_submodules"):
        gm.delete_all_unused_submodules()

    with torch.inference_mode():
        y_original = original(*example_args)
        y_packed = gm(*example_args)

    rel_l2, verified = s2_compare_outputs(
        y_original,
        y_packed,
    )

    info = {
        "accepted": bool(verified),
        "kind": "same_input_pack",
        "branches": int(len(branches)),
        "old_params": int(packed_params),
        "new_params": int(packed_params),
        "relative_l2": float(rel_l2),
    }

    if not verified:
        info["reason"] = "verification failed"
        return original, info

    return gm, info


def s2_find_resultant_fanouts(gm):
    candidates = []

    for node in gm.graph.nodes:
        if not s2_is_linear(gm, node):
            continue

        users = list(node.users)

        if (
            len(users) >= 2
            and all(s2_is_linear(gm, user) for user in users)
        ):
            candidates.append((node, users))

    return candidates


def s2_resultant_candidate(
    gm,
    upstream,
    branches,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    up = s2_get_module(gm, upstream)
    modules = [s2_get_module(gm, node) for node in branches]

    if (
        not isinstance(up, nn.Linear)
        or len(branches) < 2
        or not all(isinstance(m, nn.Linear) for m in modules)
    ):
        return None, {
            "accepted": False,
            "kind": "resultant_fanout",
            "reason": "pattern mismatch",
        }

    # Strong consumer rule:
    # upstream can disappear only if every direct consumer is one of
    # the branches being composed.
    if set(upstream.users) != set(branches):
        return None, {
            "accepted": False,
            "kind": "resultant_fanout",
            "reason": "external consumer safety",
        }

    old_params = (
        sum(p.numel() for p in up.parameters())
        + sum(
            sum(p.numel() for p in module.parameters())
            for module in modules
        )
    )

    total_out = sum(m.out_features for m in modules)

    # Resultant packed matrix:
    # [Wq A; Wk A; Wv A]
    new_params = (
        total_out * up.in_features
        + total_out
    )

    if new_params > old_params * (1.0 + max_param_growth):
        return None, {
            "accepted": False,
            "kind": "resultant_fanout",
            "reason": "cost guard",
            "old_params": int(old_params),
            "new_params": int(new_params),
        }

    new_bytes = new_params * up.weight.element_size()

    if new_bytes > max_resultant_bytes:
        return None, {
            "accepted": False,
            "kind": "resultant_fanout",
            "reason": "materialization byte guard",
            "new_bytes": int(new_bytes),
        }

    # Compose in FP32, then cast back to runtime dtype.
    W0 = up.weight.detach().float()

    if up.bias is None:
        b0 = torch.zeros(
            up.out_features,
            device=W0.device,
        )
    else:
        b0 = up.bias.detach().float()

    resultant_weights = []
    resultant_biases = []

    for module in modules:
        Wi = module.weight.detach().float()

        if module.bias is None:
            bi = torch.zeros(
                module.out_features,
                device=Wi.device,
            )
        else:
            bi = module.bias.detach().float()

        resultant_weights.append(Wi @ W0)
        resultant_biases.append(Wi @ b0 + bi)

    W = torch.cat(
        resultant_weights,
        dim=0,
    ).to(up.weight.dtype)

    b = torch.cat(
        resultant_biases,
        dim=0,
    ).to(up.weight.dtype)

    return (
        upstream.args[0],
        W,
        b,
        [m.out_features for m in modules],
    ), {
        "accepted": True,
        "kind": "resultant_fanout",
        "old_params": int(old_params),
        "new_params": int(new_params),
        "branches": int(len(branches)),
    }


def s2_apply_resultant_fanout(
    gm,
    upstream,
    branches,
    example_args,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    original = copy.deepcopy(gm)

    candidate, info = s2_resultant_candidate(
        gm,
        upstream,
        branches,
        max_param_growth=max_param_growth,
        max_resultant_bytes=max_resultant_bytes,
    )

    if candidate is None:
        return gm, info

    source, W, b, split_sizes = candidate

    name = s2_unique_name(
        gm,
        "_stage2_resultant",
    )

    gm.add_submodule(
        name,
        Stage2PackedLinear(W, b),
    )

    graph = gm.graph

    with graph.inserting_before(upstream):
        packed = graph.call_module(
            name,
            args=(source,),
        )

    with graph.inserting_after(packed):
        split = graph.call_function(
            torch.split,
            args=(packed, split_sizes),
            kwargs={"dim": -1},
        )

    previous = split
    replacements = []

    for index in range(len(branches)):
        with graph.inserting_after(previous):
            item = graph.call_function(
                operator.getitem,
                args=(split, index),
            )

        previous = item
        replacements.append(item)

    for old_node, new_node in zip(branches, replacements):
        old_node.replace_all_uses_with(new_node)

    for node in branches:
        graph.erase_node(node)

    graph.erase_node(upstream)

    graph.eliminate_dead_code()
    gm.recompile()

    if hasattr(gm, "delete_all_unused_submodules"):
        gm.delete_all_unused_submodules()

    with torch.inference_mode():
        y_original = original(*example_args)
        y_resultant = gm(*example_args)

    rel_l2, verified = s2_compare_outputs(
        y_original,
        y_resultant,
    )

    info["relative_l2"] = float(rel_l2)

    if not verified:
        info["accepted"] = False
        info["reason"] = "verification failed"
        return original, info

    return gm, info



# ============================================================
# 43. UNIVERSAL CORE: SEPARATE RESULTANT PROPAGATION
#     No fan-out packing.
# ============================================================

def s2_core_resultant_candidate(
    gm,
    upstream,
    branches,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    up = s2_get_module(gm, upstream)
    modules = [s2_get_module(gm, node) for node in branches]

    if (
        not isinstance(up, nn.Linear)
        or len(branches) < 1
        or not all(isinstance(m, nn.Linear) for m in modules)
    ):
        return None, {
            "accepted": False,
            "kind": "resultant_propagation",
            "reason": "pattern mismatch",
        }

    if set(upstream.users) != set(branches):
        return None, {
            "accepted": False,
            "kind": "resultant_propagation",
            "reason": "external consumer safety",
        }

    if any(len(node.args) < 1 or node.args[0] is not upstream for node in branches):
        return None, {
            "accepted": False,
            "kind": "resultant_propagation",
            "reason": "non-direct branch input",
        }

    old_params = (
        sum(p.numel() for p in up.parameters())
        + sum(sum(p.numel() for p in m.parameters()) for m in modules)
    )

    new_params = 0
    branch_specs = []
    for m in modules:
        need_bias = (up.bias is not None) or (m.bias is not None)
        p = m.out_features * up.in_features + (m.out_features if need_bias else 0)
        new_params += p
        branch_specs.append((m.out_features, need_bias))

    if new_params > old_params * (1.0 + max_param_growth):
        return None, {
            "accepted": False,
            "kind": "resultant_propagation",
            "reason": "cost guard",
            "old_params": int(old_params),
            "new_params": int(new_params),
        }

    new_bytes = new_params * up.weight.element_size()
    if new_bytes > max_resultant_bytes:
        return None, {
            "accepted": False,
            "kind": "resultant_propagation",
            "reason": "materialization byte guard",
            "new_bytes": int(new_bytes),
        }

    W0 = up.weight.detach().float()
    if up.bias is None:
        b0 = torch.zeros(up.out_features, device=W0.device, dtype=torch.float32)
    else:
        b0 = up.bias.detach().float()

    resultants = []
    for node, m, (_, need_bias) in zip(branches, modules, branch_specs):
        Wi = m.weight.detach().float()
        if m.bias is None:
            bi = torch.zeros(m.out_features, device=Wi.device, dtype=torch.float32)
        else:
            bi = m.bias.detach().float()

        Wr = (Wi @ W0).to(up.weight.dtype)
        br = (Wi @ b0 + bi).to(up.weight.dtype) if need_bias else None
        resultants.append({"old_node": node, "weight": Wr, "bias": br})

    return (upstream.args[0], resultants), {
        "accepted": True,
        "kind": "resultant_propagation",
        "old_params": int(old_params),
        "new_params": int(new_params),
        "branches": int(len(branches)),
        "packed": False,
    }


def s2_core_apply_resultant(
    gm,
    upstream,
    branches,
    example_args,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    original = copy.deepcopy(gm)
    candidate, info = s2_core_resultant_candidate(
        gm,
        upstream,
        branches,
        max_param_growth=max_param_growth,
        max_resultant_bytes=max_resultant_bytes,
    )
    if candidate is None:
        return gm, info

    source, resultants = candidate
    graph = gm.graph
    new_nodes = []

    for index, item in enumerate(resultants):
        Wr, br = item["weight"], item["bias"]
        module = nn.Linear(
            Wr.shape[1], Wr.shape[0],
            bias=(br is not None),
            device=Wr.device,
            dtype=Wr.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(Wr)
            if br is not None:
                module.bias.copy_(br)
        module.requires_grad_(False)

        name = s2_unique_name(gm, f"_stage2_core_resultant_{index}")
        gm.add_submodule(name, module)
        with graph.inserting_before(upstream):
            new_nodes.append(graph.call_module(name, args=(source,)))

    for item, new_node in zip(resultants, new_nodes):
        item["old_node"].replace_all_uses_with(new_node)
    for item in resultants:
        graph.erase_node(item["old_node"])
    graph.erase_node(upstream)

    graph.eliminate_dead_code()
    gm.recompile()
    if hasattr(gm, "delete_all_unused_submodules"):
        gm.delete_all_unused_submodules()

    with torch.inference_mode():
        y_original = original(*example_args)
        y_reduced = gm(*example_args)
    rel, verified = s2_compare_outputs(y_original, y_reduced)
    info["relative_l2"] = float(rel)

    if not verified:
        info["accepted"] = False
        info["reason"] = "verification failed"
        return original, info
    return gm, info


def s2_core_find_resultant_candidates(gm):
    candidates = []
    for node in gm.graph.nodes:
        if not s2_is_linear(gm, node):
            continue
        users = list(node.users)
        if len(users) >= 1 and all(s2_is_linear(gm, u) for u in users):
            candidates.append((node, users))
    return candidates



# ============================================================
# 44. UNIVERSAL STAGE-2 CORE COMPILER
#     No same-input packing of any kind.
# ============================================================

class UniversalStage2CoreCompiler:
    def __init__(
        self,
        model,
        example_args,
        max_param_growth=0.0,
        max_resultant_bytes=512 * 1024**2,
        max_passes=32,
    ):
        self.original = copy.deepcopy(model).eval()
        self.gm = stage2_symbolic_trace(model)
        self.example_args = tuple(example_args)
        self.max_param_growth = float(max_param_growth)
        self.max_resultant_bytes = int(max_resultant_bytes)
        self.max_passes = int(max_passes)
        self.log = []
        ShapeProp(self.gm).propagate(*self.example_args)

    def analyze(self):
        ShapeProp(self.gm).propagate(*self.example_args)
        barriers = [n for n in self.gm.graph.nodes if s2_is_barrier(self.gm, n)]
        resultants = s2_core_find_resultant_candidates(self.gm)
        fanouts = s2_find_same_input_fanouts(self.gm, linear_like=False)
        return {
            "mode": "universal_core",
            "barriers": [n.name for n in barriers],
            "resultant_candidates": [
                {"upstream": u.name, "branches": [b.name for b in bs]}
                for u, bs in resultants
            ],
            "same_input_fanout_annotations": [
                {
                    "source": s.name,
                    "branches": [b.name for b in bs],
                    "action": "ANNOTATE_ONLY_NO_PACKING",
                }
                for s, bs in fanouts
            ],
        }

    def _run_boundary_passes(self):
        for _ in range(self.max_passes):
            ShapeProp(self.gm).propagate(*self.example_args)
            barriers = [n for n in self.gm.graph.nodes if s2_is_barrier(self.gm, n)]
            changed = False
            for boundary in reversed(barriers):
                candidate, _ = s2_boundary_candidate(
                    self.gm,
                    boundary,
                    self.example_args,
                    max_param_growth=self.max_param_growth,
                    max_resultant_bytes=self.max_resultant_bytes,
                )
                if candidate is None:
                    continue
                self.gm, info = s2_apply_boundary(
                    self.gm,
                    boundary,
                    self.example_args,
                    max_param_growth=self.max_param_growth,
                    max_resultant_bytes=self.max_resultant_bytes,
                )
                self.log.append(info)
                if info.get("accepted"):
                    changed = True
                    break
            if not changed:
                break

    def _run_resultant_passes(self):
        for _ in range(self.max_passes):
            candidates = s2_core_find_resultant_candidates(self.gm)
            changed = False
            for upstream, branches in candidates:
                candidate, _ = s2_core_resultant_candidate(
                    self.gm,
                    upstream,
                    branches,
                    max_param_growth=self.max_param_growth,
                    max_resultant_bytes=self.max_resultant_bytes,
                )
                if candidate is None:
                    continue
                self.gm, info = s2_core_apply_resultant(
                    self.gm,
                    upstream,
                    branches,
                    self.example_args,
                    max_param_growth=self.max_param_growth,
                    max_resultant_bytes=self.max_resultant_bytes,
                )
                self.log.append(info)
                if info.get("accepted"):
                    changed = True
                    break
            if not changed:
                break

    def optimize(self):
        before = self.analyze()
        self._run_boundary_passes()
        self._run_resultant_passes()
        accepted = [r for r in self.log if r.get("accepted")]

        # A true no-op deploys the original model, not an FX wrapper.
        if not accepted:
            return self.original, {
                "mode": "universal_core",
                "verified": True,
                "relative_l2": 0.0,
                "no_op": True,
                "rewrites": [],
                "analysis_before": before,
                "analysis_after": before,
            }

        with torch.inference_mode():
            y_original = self.original(*self.example_args)
            y_optimized = self.gm(*self.example_args)
        rel, verified = s2_compare_outputs(y_original, y_optimized)

        return self.gm, {
            "mode": "universal_core",
            "verified": bool(verified),
            "relative_l2": float(rel),
            "no_op": False,
            "rewrites": accepted,
            "analysis_before": before,
            "analysis_after": self.analyze(),
        }


# ============================================================
# STAGE-2G TRANSACTIONAL REWRITE HOTFIX
# ============================================================

# Keep references to the exact original mutating implementations embedded above.
_s2_apply_boundary_mutating = s2_apply_boundary
_s2_core_apply_resultant_mutating = s2_core_apply_resultant


def _s2_node_by_name(gm, node_name):
    for node in gm.graph.nodes:
        if node.name == node_name:
            return node
    raise KeyError(f"FX node {node_name!r} not found in cloned graph")


def s2_apply_boundary(
    gm,
    boundary,
    example_args,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    """
    Transactional wrapper around the frozen Stage-2G boundary rewrite.

    A rejected trial must NOT replace the active graph, otherwise FX Node
    references collected by the current pass become stale.
    """
    trial = copy.deepcopy(gm)
    trial_boundary = _s2_node_by_name(trial, boundary.name)

    trial_result, info = _s2_apply_boundary_mutating(
        trial,
        trial_boundary,
        example_args,
        max_param_growth=max_param_growth,
        max_resultant_bytes=max_resultant_bytes,
    )

    if info.get("accepted"):
        return trial_result, info

    # Rollback means retain the exact active graph object.
    return gm, info


def s2_core_apply_resultant(
    gm,
    upstream,
    branches,
    example_args,
    max_param_growth=0.0,
    max_resultant_bytes=512 * 1024**2,
):
    """
    Transactional wrapper around the frozen Stage-2G resultant rewrite.
    """
    trial = copy.deepcopy(gm)

    trial_upstream = _s2_node_by_name(trial, upstream.name)
    trial_branches = [
        _s2_node_by_name(trial, branch.name)
        for branch in branches
    ]

    trial_result, info = _s2_core_apply_resultant_mutating(
        trial,
        trial_upstream,
        trial_branches,
        example_args,
        max_param_growth=max_param_growth,
        max_resultant_bytes=max_resultant_bytes,
    )

    if info.get("accepted"):
        return trial_result, info

    # Failed numerical verification/cost checks leave active graph untouched.
    return gm, info

def _stage2g_public_reduce(model, sample=None):
    """
    Notebook-session implementation of the planned public API:

        model = ElasticBit.reduce(model, sample)

    Uses the frozen UniversalStage2CoreCompiler from the research notebook.
    """
    if sample is None:
        raise ValueError("ElasticBit.reduce(model, sample) requires one representative sample.")

    if isinstance(sample, tuple):
        example_args = sample
    elif isinstance(sample, list):
        example_args = tuple(sample)
    else:
        example_args = (sample,)

    compiler = UniversalStage2CoreCompiler(
        model,
        example_args,
        max_param_growth=0.0,
        max_resultant_bytes=512 * 1024**2,
        max_passes=32,
    )

    reduced_model, report = compiler.optimize()

    # Store report for inspection without changing inference behavior.
    reduced_model._elasticbit_reduce_report = report
    return reduced_model


__all__ = [
    "UniversalStage2CoreCompiler",
    "_stage2g_public_reduce",
    "STAGE2G_NOTEBOOK_SOURCE",
    "STAGE2G_NOTEBOOK_CELL_SHA256",
]
