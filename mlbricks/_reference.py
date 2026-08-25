from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.ones(
        length, length, device=device, dtype=torch.bool
    ).tril()


def attention_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    # q/k/v: [B,H,T,D]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        T = q.size(-2)
        S = k.size(-2)
        if T != S:
            # Generic autoregressive alignment: last T queries attend to
            # the prefix ending at their respective positions.
            q_pos = torch.arange(S - T, S, device=q.device)[:, None]
            k_pos = torch.arange(S, device=q.device)[None, :]
            mask = k_pos <= q_pos
        else:
            mask = causal_mask(T, q.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    p = F.dropout(p, p=dropout_p, training=training)
    return torch.matmul(p, v)


def gauss_forward_reference(
    q: torch.Tensor,
    c: torch.Tensor,
    *,
    head_dim: int,
    eps: float,
    causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    # q/c: [B,H,T,R]. Key-only RMS normalization; retrieve raw C.
    rho = torch.rsqrt(c.float().square().mean(dim=-1) + eps)
    scores = torch.matmul(q, c.transpose(-2, -1))
    scores = scores * rho.unsqueeze(-2)
    scores = scores * (1.0 / math.sqrt(float(head_dim)))

    if causal:
        T = q.size(-2)
        S = c.size(-2)
        if T != S:
            q_pos = torch.arange(S - T, S, device=q.device)[:, None]
            k_pos = torch.arange(S, device=q.device)[None, :]
            mask = k_pos <= q_pos
        else:
            mask = causal_mask(T, q.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    p = F.dropout(p, p=dropout_p, training=training)
    return torch.matmul(p, c)
