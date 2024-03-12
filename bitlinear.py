"""
Implementation of the BitLinear layer described in the papers:

1. "BitNet: Scaling 1-bit Transformers for Large Language Models"
2. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"

References:
- https://arxiv.org/abs/2310.11453
- https://arxiv.org/abs/2402.17764
"""

#!/usr/bin/env python3
# Copyright (C) 2024 Charles O. Goddard

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator."""
    return x0 + (x - x0).detach()


def _quant_absmax(
    x: torch.Tensor, eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x.abs().max(dim=-1, keepdim=True).values
    x_q = (x * 128) / x_absmax.clamp(min=eps)
    x_q = x_q.clip(-128 + eps, 127 - eps).to(torch.int8)

    return x_q, x_absmax


def _quant_roundclip(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    gamma = x.abs().mean(dim=-1, keepdim=True)
    wp = x / (gamma + eps)
    wp = wp.round().clamp(-1, 1).to(torch.int8)
    return wp, gamma


def _grouped_shape(x: torch.Tensor, is_weight: bool, group_size: Optional[int] = None):
    if not group_size:
        return x.shape

    # weight:
    # (out_features, in_features) -> (group_size, -1, in_features)
    # (out_features) -> (group_size, -1)
    if is_weight:
        return tuple([group_size, -1] + list(x.shape[1:]))

    # activation:
    # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, group_size, -1)
    # (batch_size, hidden_size) -> (batch_size, group_size, -1)
    return tuple(list(x.shape[:-1]) + [group_size, -1])


@torch.compile()
def quantize(
    x: torch.Tensor,
    eps: float = 1e-9,
    mode: Literal["absmax", "roundclip"] = "absmax",
    is_weight: bool = False,
    group_size: Optional[int] = 128,
):
    xp = x.view(*_grouped_shape(x, is_weight, group_size))
    if mode == "absmax":
        res = _quant_absmax(xp, eps=eps)
    elif mode == "roundclip":
        res = _quant_roundclip(xp, eps=eps)
    else:
        raise ValueError(f"Unknown quantization mode {mode}")

    x0_scaled = (xp / res[1]).reshape_as(x)
    x_q = res[0].reshape_as(x)
    return _ste(x_q, x0_scaled), res[1]


class BitLinear(nn.Linear):
    def __init__(
        self,
        *args,
        elementwise_affine: bool = False,
        preserve_scale: bool = False,
        weight_quantization: Literal["absmax", "roundclip"] = "roundclip",
        act_quantization: Literal["absmax", "roundclip"] = "absmax",
        group_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_norm = nn.LayerNorm(
            self.in_features, elementwise_affine=elementwise_affine
        )
        self.preserve_scale = preserve_scale
        self.weight_quantization = weight_quantization
        self.act_quantization = act_quantization
        self.group_size = group_size

        if group_size:
            if preserve_scale:
                raise ValueError("Cannot preserve scale with group size")

            if self.in_features % group_size != 0:
                raise ValueError("Input size must be divisible by group size")

            if self.out_features % group_size != 0:
                raise ValueError("Output size must be divisible by group size")

    def quantized_weights(self, eps: float = 1e-9):
        if self.bias is not None:
            bias_q, _ = quantize(
                self.bias,
                eps=eps,
                mode=self.weight_quantization,
                is_weight=True,
                group_size=self.group_size,
            )
        else:
            bias_q = None

        w_q, beta = quantize(
            self.weight,
            eps=eps,
            mode=self.weight_quantization,
            is_weight=True,
            group_size=self.group_size,
        )
        return w_q, bias_q, beta

    @torch.compile()
    def forward(self, x: torch.Tensor, eps: float = 1e-9):
        x = self.input_norm(x)
        x_q, gamma = quantize(
            x,
            eps=eps,
            mode=self.act_quantization,
            is_weight=False,
            group_size=self.group_size,
        )

        w_q, bias_q, beta = self.quantized_weights(eps=eps)
        y_q = F.linear(x_q, w_q, bias_q)

        scale = beta if self.preserve_scale else 1
        y = y_q * scale / 128
        return y
