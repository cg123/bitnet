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
    x: Optional[torch.Tensor],
    is_activation: bool,
    eps: float,
    conv: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x is None:
        return None, None

    if is_activation:
        if conv:
            # (batch_sz, hidden_sz, ...)
            absmax: torch.Tensor = (
                x.abs().view(x.shape[0], -1).max(dim=-1, keepdim=True).values
            )
            while len(absmax.shape) < len(x.shape):
                absmax = absmax.unsqueeze(-1)
        else:
            # (batch_sz, seq_len, hidden_sz)
            absmax = x.abs().max(dim=-1, keepdim=True).values
        scale = 127.0 / absmax.clamp_(min=eps)
        x_q = (x * scale).round().clamp_(-128, 127)
    else:
        scale = 1.0 / x.abs().mean().clamp_(min=eps)
        x_q = (x * scale).round().clamp_(-1, 1)

    return _ste(x_q, x), scale


class QuantizedWeights(NamedTuple):
    """Quantized weight and optional bias tensor for BitLinear."""

    w_q: torch.Tensor
    bias_q: Optional[torch.Tensor]
    beta: torch.Tensor


@torch.compile()
def _quantize_weights(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
) -> QuantizedWeights:
    w_q, beta = _quantize(weight, is_activation=False, eps=eps, conv=False)
    bias_q, _ = _quantize(bias, is_activation=True, eps=eps, conv=False)
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
        preserve_scale: bool = True,
        eps: float = 1e-5,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, out_features, *args, bias=bias, **kwargs)
        self.input_norm = nn.LayerNorm(self.in_features, elementwise_affine=False)
        self.preserve_scale = preserve_scale
        self.eps = eps

    @torch.compile()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        x_q, gamma = _quantize(x, is_activation=True, eps=self.eps, conv=False)
        w_q, bias_q, beta = _quantize_weights(self.weight, self.bias, eps=self.eps)

        y = F.linear(x_q, w_q, bias_q)
        y = y.to(x.dtype)
        if self.preserve_scale:
            y = y / (gamma * beta)
        else:
            y = y / 127

        return y


class BitConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *args,
        preserve_scale: bool = True,
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
        x_q, gamma = _quantize(x, is_activation=True, eps=self.eps, conv=True)
        w_q, bias_q, beta = _quantize_weights(self.weight, self.bias, eps=self.eps)

        y = F.conv2d(x_q, w_q, bias_q, self.stride, self.padding, self.dilation)
        y = y.to(x.dtype)
        if self.preserve_scale:
            y = y / (gamma * beta)
        else:
            y = y / 127

        return y


@torch.no_grad()
def init_bitnet(module: nn.Module):
    if isinstance(module, (BitLinear, BitConv2d)):
        # d = torch.distributions.Laplace(0, 1 / math.sqrt(2))
        # module.weight[...] = (
        #     d.sample(sample_shape=module.weight.shape).reshape_as(module.weight) * 0.02
        # )
        module.weight.normal_(0, 0.02)
