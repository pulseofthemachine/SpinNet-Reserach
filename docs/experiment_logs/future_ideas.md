# Future Research Ideas

Ideas and hypotheses for future exploration.

---

## From gpt-oss (OpenAI's Open-Weight Release)
*Reference: https://github.com/openai/gpt-oss*

- [ ] **Learned Attention Sinks**: Per-head learnable bias in softmax denominator. Could combine with octonion head mixer for structured attention.
- [ ] **Banded/Sliding Window Attention**: Reduce compute for long sequences. Crucial for ICP instruction limits.
- [ ] **MXFP4 Quantization**: Study their 4-bit float approach for ternary optimization ideas.
- [ ] **Fused SwiGLU in Matmul**: Their `matmul_ogs` with `FusedActivation` - adapt for ternary.
- [ ] **triton_kernels Library**: Use their infrastructure for SpinNet kernels.

---

## 🎯 Priority Roadmap

| Priority | Feature | Impact | Effort | Status |
|----------|---------|--------|--------|--------|
| 🥇 | **Learned Attention Sinks** | Med | Low | Approved |
| 🥈 | **Fused Ternary Matmul** | High | Med | Backlog |
| 🥉 | **Banded Attention** | Med | Low | Approved (for long inference) |
| 4 | **Octonion Mixer Fusion** | High | High | Backlog |
| 5 | **triton_kernels adoption** | Med | Med | Research |

---

## Architecture Ideas
- [ ] **Octonion Head Mixer optimization**: Fuse 8 matmuls into batched gemm or Triton kernel
- [ ] **Quaternion vs Octonion**: Is 4D sufficient for language? Does 8D provide measurable benefits?
- [ ] **Sedenion (16D)**: Would even higher-dimensional algebras help?
- [ ] **Selective Octonion Layers**: Only apply octonion mixing to layers that use it (track beta/W norms, remove from layers where e₀ dominates)
- [ ] **Asymmetric Dimensions**: Give e₀ more capacity (256 dims) vs e₁-e₇ (32 each) - but requires algebra rethink

## Training Experiments
- [ ] **Full 20-epoch TinyStories**: Match baseline training budget, compare final loss
- [ ] **Scaling laws**: How does octonion weight-sharing affect scaling behavior?
- [ ] **Cross-domain transfer**: Train on code, test on language (and vice versa)

## Inference Experiments
- [ ] **Dimension ablation**: Zero out e₅ or e₇ - does coherence break?
- [ ] **Sparse inference**: Skip zero weights entirely at inference time

## Analysis
- [ ] **Gradient flow per dimension**: Which dimensions have highest gradient norms?
- [ ] **Layer-wise specialization**: Do early vs. late layers use dimensions differently?
- [ ] **Token-level probing**: What activates specific dimensions for specific tokens?
- [ ] **CHILDES comparison**: Compare model errors to child language acquisition corpora

---

## Quick Reference

### Tools
- `tools/analyze_octonion.py` - Dimension sparsity and activation analysis
- `tools/load_spinnet.py` - Load .spinnet files in PyTorch
- `compress.py` - Convert PyTorch checkpoint to .spinnet

### Key Files
- `src/model/chassis.py` - OctonionHeadMixer implementation
- `src/model/physics.py` - OctonionTernaryLinear layer
- `inference/src/model.rs` - Rust inference engine
