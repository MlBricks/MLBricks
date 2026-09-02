# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE.md and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
# Source builds can include several C++/CUDA extensions. Keep the default
# compiler fan-out conservative so editable/source installs do not exhaust
# notebook RAM. Users can override MAX_JOBS explicitly.
os.environ.setdefault("MAX_JOBS", "1")
force_cpu = os.getenv("MLBRICKS_FORCE_CPU", "0") == "1"
force_cuda = os.getenv("MLBRICKS_FORCE_CUDA", "0") == "1"
has_cuda_toolkit = (
    CUDA_HOME is not None
    and not force_cpu
    and (torch.version.cuda is not None or force_cuda)
)


def _extension(
    name: str,
    sources: list[Path],
    *,
    include_dirs: list[Path] | None = None,
    cuda_source: Path | None = None,
    use_fast_math: bool = False,
    lineinfo: bool = False,
    libraries: list[str] | None = None,
):
    # setuptools requires extension source paths to be relative to setup.py.
    # Absolute paths can leak into egg-info/SOURCES.txt and make wheel builds
    # fail even when the source files exist.
    def _rel(path: Path) -> str:
        return path.resolve().relative_to(ROOT).as_posix()

    source_strings = [_rel(path) for path in sources]
    define_macros = []
    extra_compile_args: dict[str, list[str]] = {"cxx": ["-O3", "-std=c++17"]}
    extension_cls = CppExtension

    if has_cuda_toolkit and cuda_source is not None:
        extension_cls = CUDAExtension
        source_strings.append(_rel(cuda_source))
        define_macros.append(("WITH_CUDA", None))
        nvcc = ["-O3"]
        if use_fast_math:
            nvcc.append("--use_fast_math")
        if lineinfo:
            nvcc.append("-lineinfo")
        extra_compile_args["nvcc"] = nvcc

    kwargs = dict(
        name=name,
        sources=source_strings,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    if include_dirs:
        kwargs["include_dirs"] = [_rel(path) for path in include_dirs]
    if libraries and extension_cls is CUDAExtension:
        kwargs["libraries"] = libraries

    return extension_cls(**kwargs)


def enabled(name: str) -> bool:
    return os.getenv(name, "1") != "0"


extensions = []

if enabled("MLBRICKS_BUILD_CORE_NATIVE"):
    csrc = ROOT / "mlbricks" / "bolt"
    extensions.append(
        _extension(
            "mlbricks._C",
            [csrc / "ops.cpp", csrc / "ops_cpu.cpp"],
            cuda_source=csrc / "ops_cuda.cu",
            use_fast_math=True,
            lineinfo=True,
            libraries=["cublas"],
        )
    )

if enabled("MLBRICKS_BUILD_BOLT_NATIVE") and has_cuda_toolkit:
    csrc = ROOT / "mlbricks" / "bolt"
    extensions.append(
        _extension(
            "mlbricks._gauss_cuda",
            [csrc / "bolt_attention_bindings.cpp"],
            cuda_source=csrc / "bolt_attention_cuda.cu",
            use_fast_math=True,
            lineinfo=True,
        )
    )

if enabled("MLBRICKS_BUILD_VESA_NATIVE"):
    csrc = ROOT / "mlbricks" / "vesa" / "csrc"
    extensions.append(
        _extension(
            "mlbricks.vesa._C",
            [csrc / "ops.cpp", csrc / "ops_cpu.cpp"],
            cuda_source=csrc / "ops_cuda.cu",
            lineinfo=True,
            libraries=["cublas"],
        )
    )


if enabled("MLBRICKS_BUILD_VISION_NATIVE"):
    csrc = ROOT / "mlbricks" / "vision_csrc"
    extensions.append(
        _extension(
            "mlbricks._vision_native",
            [csrc / "ops.cpp", csrc / "ops_cpu.cpp"],
            cuda_source=csrc / "ops_cuda.cu",
            use_fast_math=True,
            lineinfo=True,
        )
    )

if enabled("MLBRICKS_BUILD_FFNBRICK_NATIVE"):
    csrc = ROOT / "mlbricks" / "ffnbrick" / "csrc"
    extensions.append(
        _extension(
            "mlbricks.ffnbrick._C",
            [csrc / "bindings.cpp", csrc / "ffnbrick.cpp"],
            include_dirs=[csrc],
            cuda_source=csrc / "cuda" / "fused_ops.cu",
        )
    )

if enabled("MLBRICKS_BUILD_RESIDUALBRICK_NATIVE"):
    csrc = ROOT / "mlbricks" / "residualbrick" / "csrc"
    extensions.append(
        _extension(
            "mlbricks.residualbrick._C",
            [csrc / "bindings.cpp", csrc / "residualbrick.cpp"],
            include_dirs=[csrc],
            cuda_source=csrc / "cuda" / "residual_cuda.cu",
        )
    )

# ElasticBit 0.2 is a CUDA-only 4-32 bit runtime. Keep it optional so
# CPU/source installs still expose the PyTorch compatibility implementation.
if enabled("MLBRICKS_BUILD_ELASTICBIT_NATIVE") and has_cuda_toolkit:
    csrc = ROOT / "mlbricks" / "elasticbit" / "csrc"
    extensions.append(
        _extension(
            "mlbricks.elasticbit._C",
            [],
            cuda_source=csrc / "elasticbit_runtime.cu",
            use_fast_math=True,
        )
    )

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    include_package_data=True,
    zip_safe=False,
)
