# Pipeline Status (2026-05-04 20:26 CST) — RECOVERY

## Active Processes

| Step | PID | GPU | Description |
|------|-----|-----|-------------|
| Recovery | 3633473 | 0 | run_recovery.sh (Step 3 → Step 5 → metrics) |

## Fixes Applied

| Bug | File | Fix |
|-----|------|-----|
| CUDA OOM (Step 3) | `exp_llm_cond/crysllmgen_sample_cond.py:231` | batch_size: 1024 → `args.batch_size` (32) |
| Assertion error (Step 5) | `exp_llm_crossattn/crysllmgen_sample_crossattn.py:132` | `encode_llm_feature` returns squeezed `[4096]` instead of `[1,4096]` |
| smact_validity NoneType | `eval_utils.py:138` | Guard against `None` oxidation_states |
| CUDA OOM (Step 5) | `exp_llm_crossattn/crysllmgen_sample_crossattn.py:231` | batch_size: 1024 → `args.batch_size` (32) |

## Completed (before recovery)

- Step 1: Original sampling → `results/llama3_sample_mp_20_10000.pt`
- Step 2: Additive cond training → `out/mp_20/04052026/102707/model_final.pt` (epoch 500)
- Step 4: Cross-attn training → `out/mp_20/04052026/120614/model_final.pt` (epoch 500)

## Recovery Flow

```
Step 3 (additive cond sampling) on GPU 0 (~1.5-2h)
  → Step 5 (cross-attn sampling) on GPU 0 (~1.5-2h)
    → compute_metrics.py × 3
```

## Quick Commands

```bash
ps aux | grep 3633473 | grep -v grep  # check alive
tail -c 20000 run_recovery.log | tr '\r' '\n' | tail -10  # latest progress
```
