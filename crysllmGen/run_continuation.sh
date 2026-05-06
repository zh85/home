#!/bin/bash
# Continuation script: waits for Step 1 (sampling) and Step 2 (cond training),
# launches Step 4 (cross-attn training) as soon as a GPU frees up,
# then runs Step 3, Step 5, and computes all metrics.

find_best_gpu() {
    local min_mb=${1:-10000}
    local best_gpu=""
    local best_mem=0
    while IFS=, read -r idx mem_free mem_used util; do
        idx=$(echo "$idx" | xargs)
        mem_free=$(echo "$mem_free" | xargs)
        if [ "$mem_free" -gt "$best_mem" ] && [ "$mem_free" -ge "$min_mb" ]; then
            best_mem=$mem_free
            best_gpu=$idx
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    echo "$best_gpu"
}

wait_for_pid() {
    local pid=$1
    local label=$2
    echo "[$(date '+%H:%M:%S')] Waiting for $label (PID: $pid)..."
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[$(date '+%H:%M:%S')] $label finished"
}

cd /zhdd/home/hengzhang/code/crysllmgen-main

# PIDs of currently running jobs
PID1=2398858  # Step 1: Original sampling
PID2=2398859  # Step 2: Additive cond training

CHKPT="out/mp_20/03052026/211726/model_final.pt"

echo "[$(date '+%H:%M:%S')] Continuation script started"
echo "  Step 1 PID: $PID1 (original sampling)"
echo "  Step 2 PID: $PID2 (additive cond training)"

# Wait for Step 1 to finish first (it's nearest to completion)
wait_for_pid "$PID1" "Step 1 (original sampling)"

# Now Step 1 GPU is free - launch Step 4 (cross-attention training)
GPU4=$(find_best_gpu 60000)
echo "[$(date '+%H:%M:%S')] Step 4: Cross-attention training on GPU $GPU4"
CUDA_VISIBLE_DEVICES=$GPU4 python -W ignore exp_llm_crossattn/diff_train_crossattn.py \
    --dataset mp_20 --epochs 500 --batch_size 512 --use_llm_cond &
PID4=$!
echo "  Step 4 PID: $PID4 on GPU $GPU4"

# Wait for Step 2 to finish
wait_for_pid "$PID2" "Step 2 (additive cond training)"

echo "[$(date '+%H:%M:%S')] Step 1 & Step 2 completed"

# ---- Step 3: Additive conditioned sampling ----
GPU=$(find_best_gpu 16000)
echo "[$(date '+%H:%M:%S')] Step 3: Additive conditioned sampling on GPU $GPU"
COND_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f -newer "$CHKPT" 2>/dev/null | sort -r | head -1)
[ -z "$COND_CHKPT" ] && COND_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f | sort -r | head -1)
echo "Additive cond checkpoint: $COND_CHKPT"

CUDA_VISIBLE_DEVICES=$GPU python -W ignore exp_llm_cond/crysllmgen_sample_cond.py \
    --dataset mp_20 --chkpt_name "$COND_CHKPT" \
    --num_samples 10000 --batch_size 32 --use_llm_cond --out-prefix results/llama3_sample_cond

# Wait for Step 4 to finish (may still be running)
wait_for_pid "$PID4" "Step 4 (cross-attn training)"

# ---- Step 5: Cross-attention conditioned sampling ----
GPU=$(find_best_gpu 16000)
echo "[$(date '+%H:%M:%S')] Step 5: Cross-attention conditioned sampling on GPU $GPU"
CA_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f -newer "$CHKPT" 2>/dev/null | sort -r | head -1)
[ -z "$CA_CHKPT" ] && CA_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f | sort -r | head -1)
echo "Cross-attn checkpoint: $CA_CHKPT"

CUDA_VISIBLE_DEVICES=$GPU python -W ignore exp_llm_crossattn/crysllmgen_sample_crossattn.py \
    --dataset mp_20 --chkpt_name "$CA_CHKPT" \
    --num_samples 10000 --batch_size 32 --use_llm_cond --out-prefix results/llama3_sample_crossattn

echo "[$(date '+%H:%M:%S')] All done"

# ---- Compute metrics ----
echo "[$(date '+%H:%M:%S')] Computing metrics for original sampling..."
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Computing metrics for additive conditioned sampling..."
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_cond_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Computing metrics for cross-attention conditioned sampling..."
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_crossattn_mp_20_10000.pt \
    --eval_model_name mp20 --tasks gen --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Metrics done"
