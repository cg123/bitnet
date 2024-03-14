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

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator."""
    return x0 + (x - x0).detach()


@torch.compile()
def _quantize(
    x: Optional[torch.Tensor], is_input: bool, num_groups: int, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x is None:
        return None, None

    x0 = x
    if is_input:
        # split last dimension into num_groups
        x = x.view(list(x.shape[:-1]) + [num_groups, -1])
        scale_factor = x.abs().max(dim=-1, keepdim=True).values
    else:
        # first dimension is output features, so split that
        x = x.view([num_groups, -1] + list(x.shape[1:]))
        scale_factor = x.abs().mean(dim=list(range(1, len(x.shape))), keepdim=True)

    x_scaled = x / (scale_factor + eps)
    if is_input:
        x_q = (x_scaled * 127).clamp(-127, 127).to(torch.int8)
    else:
        x_q = x_scaled.round().clamp(-1, 1).to(torch.int8)

        # adjust scale_factor to match shape returned for input
        scale_factor = scale_factor.view(1, 1, num_groups, 1)

    return _ste(x_q, x_scaled).view_as(x0), scale_factor


class QuantizedWeights(NamedTuple):
    """Quantized weight and optional bias tensor for BitLinear."""

    w_q: torch.Tensor
    bias_q: Optional[torch.Tensor]
    beta: torch.Tensor


@torch.compile()
def _quantize_weights(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_groups: int,
    eps: float,
) -> QuantizedWeights:
    w_q, beta = _quantize(weight, is_input=False, num_groups=num_groups, eps=eps)
    bias_q, _ = _quantize(bias, is_input=True, num_groups=num_groups, eps=eps)
    # bias assumes the scale factor of weights
    return QuantizedWeights(w_q=w_q, bias_q=bias_q, beta=beta)


def _pack_ternary(x: torch.Tensor) -> torch.Tensor:
    """Pack ternary float tensor into int8 tensor. Uses ~1.6 bits per element."""

    x_packed = torch.empty(
        x.shape[:-1] + (math.ceil(x.shape[-1] / 5)), dtype=torch.int8
    )
    for i in range(0, x.shape[-1], 5):
        chunk = x[..., i : i + 5].to(torch.int8).view(x.shape[:-1] + (1, 5))
        # -1 -> 0, 0 -> 1, 1 -> 2
        chunk = chunk + 1
        # store as base-3 number
        chunk = (
            chunk
            * torch.tensor([1, 3, 9, 27, 81], device=chunk.device, dtype=chunk.dtype)
        ).sum(dim=-1)
        x_packed[..., i // 5] = chunk
    return x_packed


class BitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args,
        preserve_scale: bool = False,
        num_groups: int = 1,
        eps: float = 1e-7,
        bias: bool = False,
        **kwargs,
    ):
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if num_groups > 1 and out_features % num_groups != 0:
            raise ValueError("out_features must be divisible by num_groups")

        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        self.input_norm = nn.LayerNorm(self.in_features, elementwise_affine=False)
        self.preserve_scale = preserve_scale
        self.num_groups = num_groups
        self.eps = eps

    @torch.compile()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x_q, gamma = _quantize(
            x, is_input=True, num_groups=self.num_groups, eps=self.eps
        )
        w_q, bias_q, beta = _quantize_weights(
            self.weight, self.bias, num_groups=self.num_groups, eps=self.eps
        )

        y = F.linear(x_q, w_q, bias_q)
        y = y.to(x.dtype) / 127
        if self.preserve_scale:
            y_grouped = y.view(list(y.shape[:-1]) + [self.num_groups, -1])
            y = (y_grouped * gamma * beta).reshape_as(y)

        return y


class BitConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *args,
        preserve_scale: bool = False,
        eps: float = 1e-7,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, *args, bias=bias, **kwargs
        )
        self.input_norm = nn.GroupNorm(1, self.in_channels, affine=False)
        self.preserve_scale = preserve_scale
        self.eps = eps

    @torch.compile()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x_q, gamma = _quantize(x, is_input=True, num_groups=1, eps=self.eps)
        w_q, bias_q, beta = _quantize_weights(
            self.weight, self.bias, num_groups=1, eps=self.eps
        )

        y = F.conv2d(x_q, w_q, bias_q, self.stride, self.padding, self.dilation)
        y = y.to(x.dtype) / 127
        if self.preserve_scale:
            y_grouped = y.view(list(y.shape[:-1]) + [1, -1])
            y = (y_grouped * gamma * beta).reshape_as(y)

        return y
