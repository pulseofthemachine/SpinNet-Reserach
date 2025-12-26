# SpinNet (Research Preview)

**Status:** `EXPERIMENTAL` / `ACTIVE DEV`

SpinNet is an exploratory architecture combining **1.58-bit (Ternary) Quantization** (à la BitNet) with **Hyper-Complex Algebras**. We replace standard linear layers with **Octonion (8D)** or **Hadamard (32D)** multiplications, compressing the "brain" of the model by 8/32x while maintaining expressivity through structured geometric mixing.

Currently running on **CUDA** (via custom Triton kernels) and **WebAssembly** (on the Internet Computer blockchain).

---

## 🧪 The Hypothesis

Standard LLMs treat dimensions as independent. We force dimensions to interact via structured algebras:

| Algebra | Dimension | Compression | Mixing | Complexity |
|---------|-----------|-------------|--------|------------|
| **Octonion** | 8D | 1/8th params | Cayley-Dickson | O(n²) |
| **Hadamard** | 32D | 1/32th params | Fast Hadamard Transform | O(n log n) |

This allows:
1. **Extreme Compression**: 0.87M "brain" params for a 26M total model
2. **Memory Efficiency**: Ternary weights {-1, 0, +1} = 1.58 bits
3. **Fast Mixing**: FHT provides structured mixing with zero learned params

---

## 🏗️ Architecture Overview

### Algebra Selection

```python
SpinNetConfig(
    algebra="hadamard",  # or "octonion"
    head_mixing=True,    # Enable algebra-based head mixing
    n_head=32,           # Must be divisible by algebra dimension
    n_embd=512,
)
```

### Key Features

| Feature | Octonion (8D) | Hadamard (32D) |
|---------|---------------|----------------|
| Linear Compression | 8x | 32x |
| Head Groups | Groups of 8 | Groups of 32 |
| Mixing Method | Cayley-Dickson | FHT (O(n log n)) |
| Triton Kernels | ✅ Fused | ✅ FP32 accumulators |
| Inference Packing | ✅ 4x memory | ✅ 16x memory |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy triton tiktoken datasets transformers tqdm
```

### 2. Prepare Dataset
```bash
# TinyStories (recommended for testing)
python data/tinystories/prepare.py
```

### 3. Train
```bash
# Hadamard 32D (NEW - faster convergence, more compression)
python train.py config/train_tinystories_hadamard.py

# Octonion 8D (original)
python train.py config/train_tinystories_octonion.py
```

### 4. Generate
```bash
python generate.py --ckpt experiments/out-tinystories-hadamard/ckpt.pt \
    --prompt "Once upon a time" --max_tokens 100
```

---

## 📊 Current Status

### ✅ Verified Working
- **Hadamard 32D**: 0.87M brain params, loss 3.5 @ 200 iters, 25 tok/s
- **Octonion 8D**: Full training + inference pipeline
- **Head Mixing**: Both algebras support attention head mixing
- **KV Cache**: 4.6x speedup for autoregressive generation
- **ICP Deployment**: WebAssembly inference on Internet Computer

### ⚠️ Rust/Wasm Status
The Rust inference engine (`inference/`) currently supports **Octonion (8D) only**. Hadamard 32D support is not yet implemented.

```
inference/src/model.rs  - Octonion 8D ✅ | Hadamard 32D ❌
```

---

## 📂 Key Files

### Python Training & Inference
| File | Description |
|------|-------------|
| `src/model/chassis.py` | Model architecture with algebra selection |
| `src/model/fht_cuda.py` | Hadamard 32D kernels with FHT |
| `src/model/cayley_dickson_cuda.py` | Octonion 8D Triton kernels |
| `config/train_tinystories_hadamard.py` | Hadamard training config |
| `config/train_tinystories_octonion.py` | Octonion training config |

### Rust/Wasm Inference (Octonion only)
| File | Description |
|------|-------------|
| `inference/src/model.rs` | Rust inference with KV cache |
| `inference/src/tokenizer.rs` | GPT-2/char tokenizer |
| `inference/src/lib.rs` | IC Canister API |

---

## 🗺️ Roadmap

### Phase 1: Core Engine ✅
- [x] Octonion 8D linear layers + head mixer
- [x] Fused Triton kernels (6x speedup)
- [x] KV Cache for fast inference
- [x] Rust/Wasm inference engine

### Phase 2: Hadamard 32D ✅ NEW
- [x] Fast Hadamard Transform (FHT) kernel
- [x] 32D linear layers with O(n log n) mixing
- [x] Variance-preserving beta initialization
- [x] FP32 accumulators for numerical stability
- [x] Ternary weight packing (16x memory reduction)

### Phase 3: Deployment 🚧
- [ ] Hadamard support in Rust/Wasm
- [ ] Client-side browser inference
- [ ] 100M+ param model on FineWeb-Edu
- [ ] ICP mainnet deployment

---

## � Technical Details

### Variance-Preserving Initialization
All layers use `beta = sqrt(3 / (2 * fan_in))` for ternary weights, ensuring healthy gradient flow:
```python
# Ternary E[w²] ≈ 2/3, so scale = sqrt(1 / (fan_in * 2/3))
beta_init = math.sqrt(3.0 / (2.0 * in_o))
```

### Parameter Breakdown (Hadamard 32D, TinyStories)
```
Total:      26.60M
Embedding:  25.73M (97%)
Brain:       0.87M (3%)  ← The actual "reasoning" part!
```

---

## 📚 References

- [BitNet: 1-bit LLMs](https://arxiv.org/abs/2310.11453)
- [Fast Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform)
- [Cayley-Dickson Construction](https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction)