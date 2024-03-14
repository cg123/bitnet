# 7/15/2023
# Copyright (C) 2023 Charles O. Goddard

import torch
import transformers


class LlamaOffsetRope(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

        self.offset = 0  # a special treat for later

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.offset + seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=self.offset + seq_len, device=x.device, dtype=x.dtype
            )

        return (
            self.cos_cached[self.offset : self.offset + seq_len].to(dtype=x.dtype),
            self.sin_cached[self.offset : self.offset + seq_len].to(dtype=x.dtype),
        )


def llama_patch_rope_offset(model: transformers.LlamaForCausalLM, **kwargs):
    kwargs.update({"device": model.device})
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = LlamaOffsetRope(layer.self_attn.head_dim, **kwargs)


def llama_set_rope_offset(model: transformers.LlamaForCausalLM, offset: int):
    while hasattr(model, "model") and not hasattr(model, "layers"):
        model = model.model

    for layer in model.layers:
        layer.self_attn.rotary_emb.offset = offset
