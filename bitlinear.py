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
    wp = wp.round().clamp(-1, 1)
    return wp, gamma


def quantize(
    x: torch.Tensor, eps: float = 1e-9, mode: Literal["absmax", "roundclip"] = "absmax"
):
    if mode == "absmax":
        return _quant_absmax(x, eps=eps)
    elif mode == "roundclip":
        return _quant_roundclip(x, eps=eps)
    else:
        raise ValueError(f"Unknown quantization mode {mode}")


class BitLinear(nn.Linear):
    def __init__(
        self,
        *args,
        elementwise_affine: bool = False,
        preserve_scale: bool = False,
        weight_quantization: Literal["absmax", "roundclip"] = "roundclip",
        act_quantization: Literal["absmax", "roundclip"] = "absmax",
        group_size: Optional[int] = 128,
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

    def forward(self, x: torch.Tensor, eps: float = 1e-9):
        x = self.input_norm(x)
        x0 = x

        if self.group_size is not None:
            x = x.view(-1, self.group_size, x.shape[-1])
        x_q, gamma = quantize(x, eps=eps, mode=self.act_quantization)
        x_q = x_q.reshape_as(x0)

        if self.bias is not None:
            bias_q, _ = _ste(
                quantize(self.bias, eps=eps, mode=self.weight_quantization),
                self.bias.data,
            )
        else:
            bias_q = None

        w_q, beta = quantize(
            self.weight.view(self.group_size or 1, -1),
            eps=eps,
            mode=self.weight_quantization,
        )
        w_q = w_q.reshape_as(self.weight)

        y_q = F.linear(_ste(x_q, x0), _ste(w_q, self.weight), bias_q)

        scale = gamma * beta if self.preserve_scale else 1
        y = y_q * scale / 128
        return y
