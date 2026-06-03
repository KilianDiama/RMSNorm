import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  RMSNorm (fused, no dtype roundtrips)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm: dim must be positive.")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        if x.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"RMSNorm: expected last dim {self.weight.shape[0]}, got {x.shape[-1]}"
            )
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight.to(x.dtype) * x * rms


# ============================================================
#  SwiGLU (gated init)
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        if d_model <= 0 or hidden <= 0:
            raise ValueError("SwiGLU: d_model and hidden must be positive.")

        self.d_model = d_model
        self.hidden = hidden

        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight, gain=1.0)
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"SwiGLU: expected last dim {self.d_model}, got {x.shape[-1]}"
            )
        gate = F.silu(self.w1(x))
        val = self.w2(x)
        return self.w3(gate * val)


# ============================================================
#  RoPE (vectorized, no complex dtype)
# ============================================================

def precompute_rope(
    dim: int,
    max_seq: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns RoPE table of shape (max_seq, dim/2, 2) with cos/sin.
    dim must be even and corresponds to the *per-head* dimension.
    """
    if dim % 2 != 0:
        raise ValueError("RoPE: dim must be even.")
    if max_seq <= 0:
        raise ValueError("RoPE: max_seq must be positive.")

    half = dim // 2
    arange = torch.arange(0, half, device=device, dtype=dtype)
    freq = 1.0 / (theta ** (arange / half))
    t = torch.arange(max_seq, device=device, dtype=dtype)
    angles = torch.outer(t, freq)  # (max_seq, half)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (max_seq, half, 2)


def apply_rope(x: torch.Tensor, rope: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    x: (b, s, h, d)
    rope: (max_seq, d/2, 2) precomputed with dim = d
    """
    if x.dim() != 4:
        raise ValueError(f"apply_rope: expected x with 4 dims, got {x.dim()}.")
    b, s, h, d = x.shape
    if d % 2 != 0:
        raise ValueError("apply_rope: last dim of x must be even.")

    half = d // 2
    if rope.shape[1] != half:
        raise ValueError(
            f"apply_rope: rope second dim {rope.shape[1]} != half dim {half}."
        )
    if offset < 0 or offset + s > rope.shape[0]:
        raise ValueError(
            f"apply_rope: invalid offset {offset} for seq len {s} and rope len {rope.shape[0]}."
        )

    x1, x2 = x[..., :half], x[..., half:]

    rope_slice = rope[offset: offset + s]  # (s, half, 2)
    rope_slice = rope_slice.to(x.dtype).unsqueeze(0).unsqueeze(2)  # (1, s, 1, half, 2)
    cos, sin = rope_slice[..., 0], rope_slice[..., 1]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


# ============================================================
#  KV Cache (preallocated)
# ============================================================

@dataclass
class KVCache:
    k: torch.Tensor  # (b, max_seq, n_kv, d)
    v: torch.Tensor  # (b, max_seq, n_kv, d)
    index: int = 0

    @property
    def seq_len(self) -> int:
        return self.index

    @property
    def max_seq(self) -> int:
        return self.k.shape[1]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """
        k_new, v_new: (b, s, n_kv, d)
        """
        if k_new.shape != v_new.shape:
            raise ValueError("KVCache.append: k_new and v_new must have same shape.")

        b, s, n_kv, d = k_new.shape
        if b != self.k.shape[0] or n_kv != self.k.shape[2] or d != self.k.shape[3]:
            raise ValueError(
                "KVCache.append: shape mismatch between new KV and cache tensors."
            )

        if self.index + s > self.max_seq:
            raise ValueError(
                f"KVCache.append: exceeding max_seq {self.max_seq} with index {self.index} and s={s}."
            )

        self.k[:, self.index:self.index + s] = k_new
        self.v[:, self.index:self.index + s] = v_new
        self.index += s


def init_kv_cache(
    batch_size: int,
    max_seq: int,
    n_kv_heads: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> KVCache:
    k = torch.empty(batch_size, max_seq, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.empty(batch_size, max_seq, n_kv_heads, head_dim, device=device, dtype=dtype)
    return KVCache(k=k, v=v, index=0)


# ============================================================
#  Titan Attention (no repeat_interleave, grouped KV)
# ============================================================

class TitanAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("TitanAttention: d_model must be divisible by n_heads.")
        if n_heads % n_kv_heads != 0:
            raise ValueError("TitanAttention: n_heads must be divisible by n_kv_heads.")
        if n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("TitanAttention: n_heads and n_kv_heads must be positive.")

        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.group = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (b, t, n_kv, d)
        returns: (b, t, n_heads, d) by grouped expansion, no repeat_interleave.
        """
        if x.dim() != 4:
            raise ValueError(f"_expand_kv: expected 4D tensor, got {x.dim()}D.")
        b, t, n_kv, d = x.shape
        if n_kv != self.n_kv or d != self.head_dim:
            raise ValueError(
                f"_expand_kv: expected (n_kv={self.n_kv}, d={self.head_dim}), "
                f"got (n_kv={n_kv}, d={d})."
            )
        x = x.unsqueeze(3).expand(b, t, n_kv, self.group, d)
        return x.reshape(b, t, self.n_heads, d)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        is_causal: bool = True,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        x: (b, s, d_model)
        rope: precomputed RoPE table for head_dim
        """
        if x.dim() != 3:
            raise ValueError(f"TitanAttention: expected x with 3 dims, got {x.dim()}.")
        b, s, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"TitanAttention: expected last dim {self.d_model}, got {d_model}."
            )

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv, self.head_dim)

        q = apply_rope(q, rope, offset)
        k = apply_rope(k, rope, offset)

        if cache is not None:
            cache.append(k, v)
            k_all = cache.k[:, :cache.seq_len]  # (b, t, n_kv, d)
            v_all = cache.v[:, :cache.seq_len]
        else:
            k_all, v_all = k, v

        # Expand KV heads by grouped expansion (no repeat_interleave)
        k_all = self._expand_kv(k_all)  # (b, t, n_heads, d)
        v_all = self._expand_kv(v_all)

        q = q.transpose(1, 2)      # (b, n_heads, s, d)
        k_all = k_all.transpose(1, 2)  # (b, n_heads, t, d)
        v_all = v_all.transpose(1, 2)  # (b, n_heads, t, d)

        out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).reshape(b, s, self.d_model)
        return self.wo(out), cache


# ============================================================
#  Omni Recursive Learner V3 — 10/10
# ============================================================

class OmniRecursiveLearner(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        num_layers: int = 12,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("OmniRecursiveLearner: d_model must be divisible by n_heads.")
        if n_heads % n_kv_heads != 0:
            raise ValueError("OmniRecursiveLearner: n_heads must be divisible by n_kv_heads.")
        if num_layers <= 0:
            raise ValueError("OmniRecursiveLearner: num_layers must be positive.")

        device = device or torch.device("cpu")

        head_dim = d_model // n_heads
        rope = precompute_rope(head_dim, max_seq_len, theta, device=device, dtype=dtype)
        self.register_buffer("rope", rope, persistent=False)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": RMSNorm(d_model),
                "attn": TitanAttention(d_model, n_heads, n_kv_heads),
                "norm2": RMSNorm(d_model),
                "ffn": SwiGLU(d_model, int(d_model * 8 / 3)),
                "scale1": nn.Parameter(torch.ones(d_model, dtype=dtype) * 1e-2),
                "scale2": nn.Parameter(torch.ones(d_model, dtype=dtype) * 1e-2),
            })
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

    def init_caches(
        self,
        batch_size: int,
    ) -> List[KVCache]:
        """
        Utility to initialize KV caches for all layers.
        """
        return [
            init_kv_cache(
                batch_size=batch_size,
                max_seq=self.max_seq_len,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.d_model // (self.layers[0]["attn"].n_heads),
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        caches: Optional[List[Optional[KVCache]]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[Optional[KVCache]]]:
        """
        x: (b, s, d_model)
        caches: list of KVCache or None, length = num_layers
        """
        if x.dim() != 3:
            raise ValueError(f"OmniRecursiveLearner: expected x with 3 dims, got {x.dim()}.")
        b, s, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"OmniRecursiveLearner: expected last dim {self.d_model}, got {d_model}."
            )

        if caches is None:
            caches = [None] * self.num_layers
        elif len(caches) != self.num_layers:
            raise ValueError(
                f"OmniRecursiveLearner: expected {self.num_layers} caches, got {len(caches)}."
            )

        new_caches: List[Optional[KVCache]] = []

        for i, layer in enumerate(self.layers):
            cache = caches[i]
            offset = cache.seq_len if cache is not None else position_offset

            h, cache = layer["attn"](
                layer["norm1"](x),
                rope=self.rope,
                attn_mask=attn_mask,
                cache=cache,
                is_causal=is_causal,
                offset=offset,
            )

            x = x + layer["scale1"].to(x.dtype) * h
            x = x + layer["scale2"].to(x.dtype) * layer["ffn"](layer["norm2"](x))

            new_caches.append(cache)

        return self.final_norm(x), new_caches
