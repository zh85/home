#!/bin/bash
# Recovery script: re-run Step 3, Step 5, then compute all metrics
# Fixes applied:
#   - Batch size reduced from 1024 to 32 for diffusion sampling (OOM fix)
#   - encode_llm_feature squeezes [1,4096] -> [4096] (assertion fix)
#   - smact_validity guards against None oxidation_states

cd /zhdd/home/hengzhang/code/crysllmgen-main

GPU=0  # GPU 0 has 62 GiB free

COND_CHKPT="out/mp_20/04052026/102707/model_final.pt"
CA_CHKPT="out/mp_20/04052026/120614/model_final.pt"

echo "[$(date '+%H:%M:%S')] === Step 3: Additive conditioned sampling ==="
CUDA_VISIBLE_DEVICES=$GPU python -W ignore exp_llm_cond/crysllmgen_sample_cond.py \
    --dataset mp_20 --chkpt_name "$COND_CHKPT" \
    --num_samples 10000 --batch_size 32 --use_llm_cond \
    --out-prefix results/llama3_sample_cond
echo "[$(date '+%H:%M:%S')] Step 3 exit code: $?"

echo "[$(date '+%H:%M:%S')] === Step 5: Cross-attention conditioned sampling ==="
CUDA_VISIBLE_DEVICES=$GPU python -W ignore exp_llm_crossattn/crysllmgen_sample_crossattn.py \
    --dataset mp_20 --chkpt_name "$CA_CHKPT" \
    --num_samples 10000 --batch_size 32 --use_llm_cond \
    --out-prefix results/llama3_sample_crossattn
echo "[$(date '+%H:%M:%S')] Step 5 exit code: $?"

echo "[$(date '+%H:%M:%S')] === Computing metrics for original sampling ==="
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv
echo "[$(date '+%H:%M:%S')] Original metrics exit code: $?"

echo "[$(date '+%H:%M:%S')] === Computing metrics for additive conditioned sampling ==="
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_cond_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv
echo "[$(date '+%H:%M:%S')] Cond metrics exit code: $?"

echo "[$(date '+%H:%M:%S')] === Computing metrics for cross-attention conditioned sampling ==="
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_crossattn_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv
echo "[$(date '+%H:%M:%S')] Cross-attn metrics exit code: $?"

echo "[$(date '+%H:%M:%S')] === Recovery complete ==="
