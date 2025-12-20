# Octonion Attention A/B Test
**Date:** 2024-12-20

## Experiment Summary
Comparing training with/without OctonionHeadMixer on TinyCodes dataset.

## Configuration
| Setting | Value |
|---------|-------|
| Dataset | TinyCodes (6M tokens) |
| Model | 29M params |
| Architecture | 8 layers, 8 heads, 512 dim |
| Block size | 256 |
| Batch size | 32 |
| Grad accum | 4 |
| LR | 3e-3 |
| Octonion Attention | A: False, B: True |

## Results @ iter 200

| Variant | Train Loss | Val Loss | Time/iter |
|---------|------------|----------|-----------|
| **A: Without Mixer** | 4.58 | 4.62 | 1.56s |
| **B: With Mixer** | 4.19 | **4.26** | 2.30s |
| **Improvement** | -8.5% | **-7.8%** | +47% |

## Key Finding
**~8% loss reduction** with octonion head mixing on code, at cost of 47% slower training.

## Head Mixer Layer Analysis
Most active dimensions per layer:
| Layer | Top-3 Dims |
|-------|------------|
| 0 | e₃, e₆, e₀ |
| 1 | e₇, e₅, e₆ |
| 2 | e₅, e₂, e₀ |
| 3 | e₆, e₇, e₂ |
| 4 | e₀, e₄, e₅ |
| 5 | e₃, e₆, e₅ |
| 6 | e₀, e₁, e₆ |
| 7 | e₅, e₂, e₀ |

## Code Category Specialization
| Category | Most Active Dims |
|----------|------------------|
| functions | e₇, e₃, e₅ |
| returns | e₆, e₇, e₁ |
| loops | e₇, e₆, e₄ |
| conditionals | e₆, e₁, e₄ |
| variables | e₆, e₀, e₇ |
| operators | e₀, e₇, e₆ |

**Pattern:** e₆ and e₇ dominate code structure. Different from language patterns.

## Next Steps
- [ ] Run to completion (5k iters)
- [ ] Compare final loss
- [ ] Optimize head mixer kernel (reduce 47% overhead)
- [ ] Test on TinyStories with fixed config
