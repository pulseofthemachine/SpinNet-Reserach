# Future Research Ideas

Ideas and hypotheses for future exploration.

---

## Architecture Ideas
- [ ] **Octonion Head Mixer optimization**: Fuse 8 matmuls into batched gemm or Triton kernel
- [ ] **Quaternion vs Octonion**: Is 4D sufficient for language? Does 8D provide measurable benefits?
- [ ] **Sedenion (16D)**: Would even higher-dimensional algebras help?

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
