# SpinNet Research Notes

Running log of interesting findings, hypotheses, and ideas for future exploration.

---

## 2024-12-20: TinyStories Training + Octonion Analysis

### Key Results

| Metric | Value |
|--------|-------|
| Model | 28.9M params (ternary + octonion) |
| Training | 0.7 epochs (~10k iterations) |
| Val Loss | 2.68 |
| Compression | 331 MB → 50 MB (6x smaller than baseline) |

### Finding 1: Octonion Dimensions Specialize

Different octonion dimensions (e₀-e₇) activate for different linguistic categories:

| Category | Most Active Dims |
|----------|------------------|
| Nouns | e₀, e₁, e₇ |
| Verbs | e₀, e₇, e₁ |
| Pronouns | e₀, e₇, e₂ |
| Emotions | e₀, e₁, e₃ |
| Dialogue | e₀, e₂, e₁ |

**Interpretation:**
- e₀ (real component) = base representation, always highest
- e₇ = specificity/details
- e₃ = semantic/emotional content
- e₂ = dialogue structure

**This emerged without supervision.** The Cayley-Dickson algebra forces dimensions to interact structurally, and the model learns to exploit this.

**Future work:** Does this specialization strengthen with more training? Does it hold at larger scales?

---

### Finding 2: Val Loss < Train Loss (Ternary Regularization)

During early training, validation loss was consistently *lower* than training loss - the opposite of typical overfitting.

**Hypothesis:** Ternary quantization `{-1, 0, 1}` acts as strong regularization. The model can't memorize training data because it has limited capacity per weight.

**Implication:** Ternary models may be inherently resistant to overfitting. This could enable training on smaller datasets without the usual overfitting risks.

---

### Finding 3: Loss as Linguistic Developmental Stage

The model at val loss 2.68 makes errors similar to children learning to speak:
- Pronoun confusion ("She... He... They" mixed)
- Object/subject errors ("She was a very happy surprise")
- Repetitive naming for clarity
- Stream-of-consciousness logic

**Hypothesis:** Val loss maps to human language acquisition stages:
| Loss | Approx. Age |
|------|-------------|
| 3.5+ | 2-3 years |
| 2.5-3.0 | 3-4 years |
| 2.0-2.5 | 4-5 years |
| 1.5-2.0 | 5-6 years |
| <1.5 | 6+ years |

**Future work:** Systematic comparison of model errors vs. child language corpora (CHILDES database).

---

### Finding 4: Training Efficiency

Achieved coherent story output in 0.7 epochs vs. baseline's 20 epochs.

**Question:** How much does ternary constraint affect sample efficiency? Does the structured prior (octonion algebra) reduce the amount of data needed?

**Experiment to run:** Train SpinNet vs. standard transformer on same data budget, compare loss curves.

---

## Ideas for Future Exploration

### Architecture
- [ ] **Octonion Head Mixer**: Currently `octonion_attention=False` in the trained model. Need to test with it enabled.
- [ ] **Quaternion vs Octonion**: Is 4D sufficient for language? Or does 8D provide measurable benefits?
- [ ] **Sedenion (16D)**: Would even higher-dimensional algebras help?

### Training
- [ ] **Full 20-epoch run**: Match baseline training budget, compare final loss
- [ ] **Scaling laws**: How does octonion weight-sharing affect scaling behavior?
- [ ] **Different datasets**: Does dimension specialization transfer to code/math/dialogue?

### Inference
- [ ] **Dimension ablation**: What happens if we zero out e₅ or e₇? Does coherence break?
- [ ] **Sparse attention**: Could we skip zero weights entirely at inference time?

### Analysis
- [ ] **Gradient flow per dimension**: Which dimensions have highest gradient norms during training?
- [ ] **Layer-wise specialization**: Do early vs. late layers use dimensions differently?
- [ ] **Token-level probing**: What activates e₃ for specific tokens?

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
