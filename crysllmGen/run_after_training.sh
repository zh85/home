#!/bin/bash
# Auto-run pipeline after training completes.
# Step 1 (sampling), Step 2 (cond training), Step 4 (cross-attn training) run in parallel.
# Step 3 starts after Step 2 finishes.
# Step 5 starts after Step 4 finishes.

find_best_gpu() {
    # Return GPU index with most free memory (min required in MB as $1)
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

find_three_best_gpus() {
    # Return three different GPU indices with most free memory
    # $1: min MB for GPU1, $2: min for GPU2, $3: min for GPU3
    local min1=${1:-16000}
    local min2=${2:-60000}
    local min3=${3:-60000}
    local gpus=""
    gpus=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t',' -k2 -rn | cut -d',' -f1 | xargs)
    read g1 g2 g3 _ <<< "$gpus"
    echo "$g1 $g2 $g3"
}

wait_for_pid() {
    local pid=$1
    local label=$2
    echo "[$(date '+%H:%M:%S')] Waiting for $label (PID: $pid)..."
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    echo "[$(date '+%H:%M:%S')] $label finished"
}

cd /zhdd/home/hengzhang/code/crysllmgen-main

# PIDs of currently running prerequisite jobs
FEAT_PID=$(pgrep -f "extract_llama_features.py" | head -1)
DIFF_PID=$(pgrep -f "diff_train.py.*mp_20" | head -1)

echo "[$(date '+%H:%M:%S')] Pipeline started"
echo "  Feature extraction PID: $FEAT_PID"
echo "  Diffusion training PID: $DIFF_PID"

# Wait for both prerequisite jobs
if [ -n "$FEAT_PID" ]; then
    wait_for_pid "$FEAT_PID" "feature extraction"
fi
if [ -n "$DIFF_PID" ]; then
    wait_for_pid "$DIFF_PID" "diffusion training"
fi

echo "[$(date '+%H:%M:%S')] All prerequisites done"

# Find original diffusion checkpoint
CHKPT="out/mp_20/03052026/211726/model_final.pt"
[ ! -f "$CHKPT" ] && CHKPT=$(find out/mp_20 -name "model_final.pt" -type f | sort -r | head -1)
echo "Original checkpoint: $CHKPT"

mkdir -p results

# Get three distinct GPUs
read GPU1 GPU2 GPU3 <<< $(find_three_best_gpus 16000 60000 60000)
echo "GPU assignments: S1=$GPU1  S2=$GPU2  S4=$GPU3"
echo "[$(date '+%H:%M:%S')] Launching Step 1, Step 2, Step 4 in parallel..."

# ---- Step 1: Original (unconditioned) sampling (background) ----
CUDA_VISIBLE_DEVICES=$GPU1 python -W ignore crysllmgen_sample_llama3.py \
    --dataset mp_20 --chkpt_name "$CHKPT" \
    --num_samples 10000 --batch_size 32 --out-prefix results/llama3_sample &
PID1=$!
echo "  Step 1 PID: $PID1 on GPU $GPU1 (original sampling)"

# ---- Step 2: Additive conditioning training (background, needs 60GB+) ----
CUDA_VISIBLE_DEVICES=$GPU2 python -W ignore exp_llm_cond/diff_train_cond.py \
    --dataset mp_20 --epochs 500 --batch_size 512 --use_llm_cond &
PID2=$!
echo "  Step 2 PID: $PID2 on GPU $GPU2 (additive cond training)"

# ---- Step 4: Cross-attention conditioning training (background, needs 60GB+) ----
# Reuses dataset and LLM features from exp_llm_cond
CUDA_VISIBLE_DEVICES=$GPU3 python -W ignore exp_llm_crossattn/diff_train_crossattn.py \
    --dataset mp_20 --epochs 500 --batch_size 512 --use_llm_cond &
PID4=$!
echo "  Step 4 PID: $PID4 on GPU $GPU3 (cross-attn cond training)"

# Wait for all three parallel steps
wait_for_pid "$PID1" "Step 1 (original sampling)"
wait_for_pid "$PID2" "Step 2 (additive cond training)"
wait_for_pid "$PID4" "Step 4 (cross-attn training)"

echo "[$(date '+%H:%M:%S')] Step 1, 2, 4 completed"

# ---- Step 3: Additive conditioned sampling (depends on Step 2) ----
GPU=$(find_best_gpu 16000)
echo "[$(date '+%H:%M:%S')] Step 3: Additive conditioned sampling on GPU $GPU"
COND_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f -newer "$CHKPT" 2>/dev/null | sort -r | head -1)
# Fallback: find the second-newest checkpoint (Step 2's output, not Step 4's)
[ -z "$COND_CHKPT" ] && COND_CHKPT=$(find out/mp_20 -name "model_final.pt" -type f | sort -r | head -2 | tail -1)
echo "Additive cond checkpoint: $COND_CHKPT"

CUDA_VISIBLE_DEVICES=$GPU python -W ignore exp_llm_cond/crysllmgen_sample_cond.py \
    --dataset mp_20 --chkpt_name "$COND_CHKPT" \
    --num_samples 10000 --batch_size 32 --use_llm_cond --out-prefix results/llama3_sample_cond

# ---- Step 5: Cross-attention conditioned sampling (depends on Step 4) ----
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
    --eval_model_name mp20 \
    --tasks gen \
    --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Computing metrics for additive conditioned sampling..."
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_cond_mp_20_10000.pt \
    --eval_model_name mp20 \
    --tasks gen \
    --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Computing metrics for cross-attention conditioned sampling..."
python -W ignore compute_metrics.py \
    --root_path results/llama3_sample_crossattn_mp_20_10000.pt \
    --eval_model_name mp20 \
    --tasks gen \
    --gt_file data/mp_20/test.csv

echo "[$(date '+%H:%M:%S')] Metrics done"
