# CrysLLMGen Experimental Results

## Pipeline: LLM + Diffusion for Crystal Structure Generation

All experiments run on MP-20 dataset, using LLaMA3-8B (fine-tuned) for text feature extraction and CSPDiffusion for structure generation.

## Results Summary

| Model | LLM | Diffusion | comp_valid | struct_valid | valid | wdist_density | wdist_num_elems | cov_recall | cov_precision |
|-------|:---:|:---------:|:----------:|:----------:|:-----:|:------------:|:---------------:|:----------:|:-------------:|
| Unconditional (baseline) | ✗ | ✓ | 86.27 | 99.93 | 86.22 | 0.10 | 0.20 | **99.15** | 99.18 |
| LLM-Only | ✓ | ✗ | 86.27 | 92.08 | 79.92 | 0.98 | 0.22 | 96.75 | 97.78 |
| Additive Cond | ✓ | ✓ | 86.27 | **99.94** | 86.22 | 3.56 | 0.20 | 53.54 | 95.95 |
| Cross-Attention Cond | ✓ | ✓ | 86.27 | 99.29 | 85.75 | 1.51 | 0.20 | 94.43 | 95.27 |
| Spatial Cross-Attn Cond | ✓ | ✓ | — | — | — | — | — | — | — |

## Key Findings

1. **Unconditional diffusion** achieves the best overall performance (cov_recall 99.15), serving as the strong baseline.
2. **Cross-Attention conditioning** (cov_recall 94.43) significantly outperforms Additive Cond (53.54), demonstrating that token-level cross-attention avoids the mode collapse caused by global vector injection.
3. **Additive Cond** suffers severe mode collapse — compressing LLM features into a single global vector forces all atoms to receive the same signal, collapsing diversity.
4. **LLM-Only** (no diffusion) has poor structural validity (92.08), confirming that diffusion refinement is essential.
5. **Spatial Cross-Attn** (training complete, sampling pending) adds 3D coordinate-based spatial bias to cross-attention, expected to further close the gap with unconditional.

## Architectures

### Additive Cond
```
LLM feature [B,4096] → MLP → [B,512] global vector → added to all atom features
```

### Cross-Attention Cond
```
LLM feature [B,4096] → reshape → [B, 8 tokens, 512 dim]
Each CSPLayer: atom features cross-attend to 8 LLM tokens
```

### Spatial Cross-Attention Cond
```
Same as Cross-Attention + 3D sinusoidal position encoding bias on attention scores
```

## Model Paths

- Additive Cond: `out/mp_20/04052026/102707/model_final.pt`
- Cross-Attention: `out/mp_20/04052026/120614/model_final.pt`
- Spatial Cross-Attn: `out/mp_20/06052026/092559/model_final.pt`

## Generated Samples

All 10,000 samples available at `results/llama3_sample_*.pt`
