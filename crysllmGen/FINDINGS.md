# crysllmGen 实验结果与关键发现 (2026-05-13 更新)

## 方法论

LLM + 扩散模型晶体结构生成，MP-20 数据集，LLaMA-3-8B + LoRA。

## 最新结果：多 Epoch 训练 P0 (2026-05-13)

### 训练对比

| 指标 | 1 epoch (旧) | 5 epoch (新) |
|------|-------------|-------------|
| Train Loss | 0.566 | **0.457** |
| 训练步数 | 848 | 4,240 |
| 训练时间 | 1.4h | 6.7h |
| GPU | A100-SXM4-80GB (GPU3) | A100 80GB PCIe (GPU4) |

### 评估结果对比 (Unconditional, n=10000)

| 指标 | 1 epoch | 5 epoch | Δ |
|------|---------|---------|-----|
| comp_valid | 86.27 | **92.59** | **+6.32** |
| struct_valid | 99.93 | 99.95 | +0.02 |
| valid | 86.22 | **92.56** | **+6.34** |
| wdist_density | 0.102 | 0.394 | +0.292 |
| wdist_num_elems | 0.204 | **0.118** | -0.086 |
| cov_recall | 99.15 | 97.13 | -2.02 |
| cov_precision | 99.18 | **99.84** | +0.66 |

### 关键发现 (P0)

1. **comp_valid 暴涨 +6.32pp** — 远超预期的 1-3pp，92.59 已超过 GT comp 基准 91.87
2. **valid +6.34pp** — 全面超越之前所有方法
3. **cov_recall 微降 -2pp** — 覆盖度略降但仍在 97%+ 优秀水平
4. **元素分布改善** — wdist_num_elems 从 0.204 降至 0.118
5. **密度分布退化** — wdist_density 从 0.102 升至 0.394，需关注

**结论：多 epoch 训练是最高投入产出比的优化。LLaMA-3-8B 只需充分训练就能达到 SOTA 水平。**

---

## 全部评估结果 (含新 5-epoch model)

| Method | comp_valid | struct_valid | valid | wdist_density | wdist_#elem | cov_recall | cov_precision |
|--------|-----------|-------------|-------|--------------|-----------|------------|---------------|
| **5-epoch Unconditional** | **92.59** | 99.95 | **92.56** | 0.394 | 0.118 | 97.13 | 99.84 |
| **GT comp + rand init** | **91.87** | 99.69 | **91.61** | 0.600 | **0.012** | **99.83** | 98.3 |
| Unconditional (1 epoch) | 86.27 | 99.93 | 86.22 | **0.102** | 0.204 | 99.15 | 99.18 |
| Cross-attn | 86.27 | 99.29 | 85.75 | 1.514 | 0.204 | 94.43 | 95.27 |
| LLM-only | 86.27 | 92.08 | 79.92 | 0.976 | 0.222 | 96.75 | 97.78 |
| Conditional (additive) | 86.27 | 99.94 | 86.22 | 3.563 | 0.204 | 53.54 | 95.95 |
| Spatial | 86.27 | 92.94 | 80.38 | 1.125 | 0.216 | 55.87 | 99.41 |
| Multi-scale (lattice-first) | 85.2 | 99.9 | 85.2 | 0.076 | 0.196 | 94.31* | 98.8 |

## 方法论洞察 (更新)

```
5-epoch LLaMA-3-8B: comp_valid=92.59 (超过 GT! ), valid=92.56
扩散上限:          cov_recall=99.83 (LLM 仍不及 GT comp)
LLM 改进方向:      保持高 comp_valid，提升 cov_recall 回到 99+
                   方案 A (DPO) / 方案 E (Wyckoff表示)
```

## 下一步

1. DPO 物理反馈微调 — 用 M3GNet/CHGNet 自动打分，对齐到稳定性
2. Wyckoff 表示 — PLaID++ 验证过，可进一步改善 comp_valid + token 效率
3. Best-of-N 推理 — 组合 DPO 模型 + 物理重排序

完整代码在 `/zhdd/home/hengzhang/code/crysllmgen-main/`。
模型权重在 `exp/llama3-8b-mp-5epoch/`。

---

## 实验卡片: 5-epoch Unconditional

### 目录结构

```
exp/llama3-8b-mp-5epoch/          # LoRA 权重
  adapter_config.json
  adapter_model.safetensors         # 2.27 GB
  checkpoint-500/ ... checkpoint-4240/   # 中间 checkpoint
results/llama3_sample_5epoch_mp_20_10000.pt   # 采样结果 (2.0 MB)
eval_5epoch.log                    # 评估日志
finetune_llama3_5epoch.log         # 训练日志
sample_5epoch.log                  # 采样日志
```

### 1. LLM 微调

**脚本**: `llm_finetune_llama3.py`

```bash
CUDA_VISIBLE_DEVICES=4 \
python -u -W ignore llm_finetune_llama3.py \
  --run-name llama3-8b-mp-5epoch \
  --data-path data/mp_20 \
  --num-epochs 5 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 2e-5 \
  --lr-scheduler cosine \
  --save-freq 500 \
  --eval-freq 200 \
  --logging-steps 5 \
  2>&1 | tee finetune_llama3_5epoch.log
```

| 参数 | 值 |
|------|-----|
| 基座模型 | Meta-Llama-3-8B (`/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B`) |
| 数据集 | mp_20 (train=27,136, val=9,047) |
| Epochs | 5 |
| 总步数 | 4,240 |
| 每设备 Batch Size | 4 |
| 梯度累积 | 8 |
| 有效 Batch Size | 32 |
| 学习率 | 2e-5 |
| LR 调度器 | cosine (warmup_ratio=0.03) |
| Weight Decay | 0.01 |
| 精度 | bfloat16 |
| LoRA Rank | 16 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| LoRA Target | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 可训练参数 | 41,943,040 / 8,072,204,288 (0.52%) |
| GPU | NVIDIA A100 80GB PCIe (GPU 4) |
| 训练时间 | 24,166s (~6.7h) |
| Train Loss | 0.4573 |

### 2. 采样 + 扩散精炼

**脚本**: `crysllmgen_sample_llama3.py`

```bash
CUDA_VISIBLE_DEVICES=4 \
python -u -W ignore crysllmgen_sample_llama3.py \
  --model_path exp/llama3-8b-mp-5epoch \
  --llama_model_path /zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B \
  --dataset mp_20 \
  --num_samples 10000 \
  --batch_size 4 \
  --temperature 1.0 \
  --top_p 0.7 \
  --diff_steps 800 \
  --timesteps 1000 \
  --run-type sample \
  --out-prefix results/llama3_sample_5epoch \
  2>&1 | tee sample_5epoch.log
```

| 参数 | 值 |
|------|-----|
| LoRA adapter | `exp/llama3-8b-mp-5epoch` |
| 扩散模型 | `out/mp_20/03052026/211726/model_final.pt` |
| 采样数量 | 10,000 |
| LLM Batch Size | 4 |
| 扩散 Batch Size | 1,024 (脚本内硬编码) |
| Temperature | 1.0 |
| Top-p | 0.7 |
| Diffusion Steps | 800 |
| Timesteps | 1,000 |
| GPU | NVIDIA A100 80GB PCIe (GPU 4) |
| 采样时间 | ~3h (LLM 生成 + 扩散精炼) |
| 输出文件 | `results/llama3_sample_5epoch_mp_20_10000.pt` |

### 3. 评估

**脚本**: `compute_metrics.py`

```bash
CUDA_VISIBLE_DEVICES=1 \
python -u -W ignore compute_metrics.py \
  --root_path results/llama3_sample_5epoch_mp_20_10000.pt \
  --tasks gen \
  --eval_model_name mp20 \
  --gt_file data/mp_20/test.csv \
  2>&1 | tee eval_5epoch.log
```

| 参数 | 值 |
|------|-----|
| 评估模型 | mp20 |
| 测试集 | `data/mp_20/test.csv` |
| GPU | GPU 1 |

### 复现 Checklist

1. 确认 conda 环境: `conda activate crysllm`
2. 确认基座模型存在: `/zhdd/home/hengzhang/models/Meta-Llama-3-8B/LLM-Research/Meta-Llama-3-8B`
3. 确认数据: `data/mp_20/train.csv, val.csv, test.csv`
4. 确认扩散模型: `out/mp_20/03052026/211726/model_final.pt`
5. 检查 GPU 显存 (需 ≥50GB 空闲)
6. 按顺序执行上述三个命令
