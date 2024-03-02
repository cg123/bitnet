# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return x0 + (x - x0).detach()


def _quant_absmax(
    x: torch.Tensor, eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_absmax = x.abs().max()
    x_q = (x * 128) / x_absmax.clamp(min=eps)
    x_q = x_q.clip(-128 + eps, 127 - eps).to(torch.int8)

    return x_q, x_absmax


class BitLinear(nn.Linear):
    def __init__(self, *args, elementwise_affine: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_norm = nn.LayerNorm(
            self.in_features, elementwise_affine=elementwise_affine
        )

    def forward(self, x: torch.Tensor, eps: float = 1e-9):
        x = self.input_norm(x)

        x_q, gamma = _quant_absmax(x, eps=eps)

        if self.bias is not None:
            bias_q, _ = _ste(_quant_absmax(self.bias, eps=eps), self.bias.data)
        else:
            bias_q = None

        w_mean = self.weight.view(-1).mean()
        w_q = (self.weight - w_mean).sign()
        beta = self.weight.view(-1).abs().sum() / self.weight.numel()

        y_q = F.linear(_ste(x_q, x), _ste(w_q, self.weight), bias_q)
        y = y_q * beta * gamma / 128
        return y


def patch_model_bitlinear(model: nn.Module, eps: float = 1e-9):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if name in ("lm_head", "embed_tokens"):
                continue
            setattr(
                model,
                name,
                BitLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            patch_model_bitlinear(module, eps=eps)

    return model
