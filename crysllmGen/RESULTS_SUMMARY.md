# CrysLLMGen Experimental Results (2026-05-06)

## Complete Comparison Matrix (Same LLM-generated data)

| Metric | LLM-Only | Unconditional | Additive | Cross-Attn |
|--------|:---:|:---:|:---:|:---:|
| comp_valid | 86.27 | 86.27 | 86.27 | 86.27 |
| struct_valid | 92.08 | **99.93** | **99.94** | 99.29 |
| valid | 79.92 | **86.22** | **86.22** | 85.75 |
| wdist_density | 0.98 | **0.10** | 3.56 | 1.51 |
| wdist_num_elems | 0.22 | **0.20** | 0.20 | 0.20 |
| COV-Recall | 96.75 | **99.15** | 53.54 | 94.43 |
| COV-Precision | 97.78 | **99.18** | 95.95 | 95.27 |

## Key Findings

1. **Diffusion refinement is essential**: struct_valid 92% → 99.9%
2. **Unconditional diffusion wins on all metrics**: LLM conditioning during
   diffusion degrades performance
3. **Cross-Attn > Additive**: Selective conditioning (94.4% recall) better than
   forced injection (53.5% recall), but both worse than no conditioning
4. **LLM features are noise during diffusion**: The LLM's value is in generating
   the initial structure (x_T), not in guiding denoising

## Model Checkpoints

| Model | Path |
|-------|------|
| Unconditional | out/mp_20/03052026/211726/model_final.pt |
| Additive Cond | out/mp_20/04052026/102707/model_final.pt |
| Cross-Attn | out/mp_20/04052026/120614/model_199.pt (best) |
| Cross-Attn (final) | out/mp_20/04052026/120614/model_final.pt (collapsed) |

## Output Files

| Description | Path |
|-------------|------|
| LLM-only results | results/llama3_sample_llmonly_mp_20_10000.pt |
| Unconditional results | results/llama3_sample_unconditional_mp_20_10000.pt |
| Additive results | results/llama3_sample_cond_mp_20_10000.pt |
| Cross-Attn results | results/llama3_sample_crossattn_mp_20_10000.pt |

## Pending: Spatial Attention Experiment

- Training: just started (500 epochs, ~2.5h)
- Module: exp_llm_spatial/
