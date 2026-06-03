🚀 OmniRecursiveLearner V3 — High‑Performance Transformer Architecture
OmniRecursiveLearner V3 is a modern, efficient, and research‑grade Transformer stack featuring:

TitanAttention (GQA without repeat_interleave)

Vectorized RoPE (no complex dtype, broadcast‑friendly)

Preallocated KV‑Cache for fast autoregressive inference

RMSNorm (fused, stable)

SwiGLU with gated initialization

LayerScale‑style residual scaling

Fully modular layer structure

PyTorch native SDPA (scaled_dot_product_attention)

This module is designed for speed, clarity, and production‑ready inference.

✨ Features
🔹 TitanAttention (Grouped KV Attention)
Efficient GQA implementation

No repeat_interleave → uses grouped expansion

Q: [B, S, n_heads, D]

K/V: [B, S, n_kv_heads, D]

Broadcast expansion to full heads

SDPA for high‑performance attention

🔹 Vectorized RoPE (no complex dtype)
Precomputed RoPE table: (max_seq, dim/2, 2)

Fast rotation using real tensors

Offset‑based indexing for KV‑cache

No dtype conversions inside the loop

🔹 Preallocated KV‑Cache
Cache structure:

k: (B, max_seq, n_kv, D)

v: (B, max_seq, n_kv, D)

In‑place append

Zero reallocation

Perfect for autoregressive decoding

🔹 RMSNorm (fused)
No dtype roundtrips

Stable for deep networks

Drop‑in replacement for LayerNorm

🔹 SwiGLU MLP
Gated initialization

w1, w2, w3 with Xavier init

Efficient silu(gate) * val fusion

🔹 LayerScale‑style residuals
Per‑layer learnable scaling

Stabilizes deep stacks

Helps training at large depth

📦 Installation
bash
pip install torch
No external dependencies required.

🧩 Usage Example
python
import torch
from omni_recursive_learner import OmniRecursiveLearner

model = OmniRecursiveLearner(
    d_model=512,
    n_heads=8,
    n_kv_heads=2,
    num_layers=12,
    max_seq_len=4096,
)

# Dummy batch
x = torch.randn(1, 128, 512)

# Initialize KV caches for autoregressive mode
caches = model.init_caches(batch_size=1)

# Forward pass
out, new_caches = model(x, caches=caches)

print(out.shape)  # -> [1, 128, 512]
🧠 Technical Overview
🔸 RoPE (vectorized)
Precomputed once at initialization

Stored as (max_seq, dim/2, 2)

Applied via real‑valued rotation

No complex dtype → faster & more stable

🔸 TitanAttention
Q/K/V projections

KV grouped expansion:

python
x.unsqueeze(3).expand(b, t, n_kv, group, d)
No memory duplication

SDPA handles causal masking and dropout

🔸 KV‑Cache
Preallocated tensors

In‑place updates

No dynamic resizing

Perfect for long‑context inference

🔸 OmniRecursiveLearner Layer Stack
Each layer contains:

norm1 → RMSNorm

attn → TitanAttention

scale1 → LayerScale

norm2 → RMSNorm

ffn → SwiGLU

scale2 → LayerScale

Final output is normalized with RMSNorm.

🧪 Autoregressive Decoding Example
python
# One token at a time
for t in range(100):
    x_t = torch.randn(1, 1, 512)
    out, caches = model(x_t, caches=caches, position_offset=t)
📁 Project Structure
Code
OmniRecursiveLearner/
│
├── omni_recursive_learner.py   # Full implementation
├── README.md                   # This file
└── LICENSE                     # UOSACL‑1.0 license
🔒 License
This project uses the UOSACL‑1.0 — Universal Open‑Source Attribution & Commercial License.

Non‑commercial use: free

Attribution: required

Commercial use: requires agreement + royalties

🧭 Roadmap
[ ] Add FlashAttention‑compatible kernels

[ ] Add FP8 / quantization‑friendly variant

[ ] Add multi‑block “OmniTransformer” wrapper

[ ] Add benchmark suite (speed & memory)

[ ] Add training script + dataset loader

🤝 Contributing
Contributions are welcome — optimizations, CUDA kernels, new attention variants, or performance benchmarks.

🔥 Summary
OmniRecursiveLearner V3 is a clean, modern, and efficient Transformer architecture featuring:

fast RoPE

grouped KV attention

preallocated KV‑cache

RMSNorm

SwiGLU

LayerScale

It is ideal for:

LLM research

inference engines

custom architectures

long‑context models

experimental reasoning systems
