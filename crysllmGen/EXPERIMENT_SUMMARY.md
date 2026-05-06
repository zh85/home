# crysllmGen 实验总结 (2026-05-06)

## 项目概述

本项目使用 LLM (Llama3-8B) + 扩散模型生成晶体结构 (crystal structure generation)，在 MP-20 数据集上对比了 5 种生成方法。

## 实验方法

| 方法 | 说明 |
|------|------|
| **LLM-only** | 纯 LLM 生成 CIF 结构，不经扩散模型 |
| **Unconditional** | 无条件扩散模型（不使用 LLM 特征） |
| **Conditional (additive)** | 扩散模型 + LLM 特征加法注入 |
| **Cross-attn** | 扩散模型 + 交叉注意力 LLM 条件化 |
| **Spatial cross-attn** | 扩散模型 + 空间交叉注意力 LLM 条件化（新增） |

## 评估结果 (mp_20, n=10000)

| Method | comp_valid | struct_valid | valid | wdist_density | wdist_#elem | cov_recall | cov_precision |
|--------|-----------|-------------|-------|--------------|------------|------------|---------------|
| LLM-only | 86.27 | 92.08 | 79.92 | 0.976 | 0.222 | 96.75 | 97.78 |
| Unconditional | 86.27 | **99.93** | **86.22** | **0.102** | 0.204 | **99.15** | **99.18** |
| Conditional | 86.27 | 99.94 | 86.22 | 3.563 | 0.204 | 53.54 | 95.95 |
| Cross-attn | 86.27 | 99.29 | 85.75 | 1.514 | 0.204 | 94.43 | 95.27 |
| Spatial | 86.27 | 92.94 | 80.38 | 1.125 | 0.216 | 55.87 | 99.41 |

## 结论

1. **无条件扩散模型 (Unconditional) 在 MP-20 上表现最好**，覆盖率召回率达 99.15%，结构有效率达 99.93%，密度分布最接近真实数据 (wdist=0.102)。

2. **LLM 条件化在所有形式下均降低了性能**：
   - Additive 和 Spatial 方法召回率大幅下降至 ~53-56%
   - Cross-attn 是 LLM-conditioned 方法中最好的 (94.43% recall)
   - Spatial cross-attn 的结构有效率 (92.94%) 低于其他扩散方法

3. **LLM-only 覆盖率很高 (96.75%)**，但结构有效率较低 (92.08%)，说明 LLM 生成的结构需要扩散模型校正。

## 文件路径

| 类别 | 路径 | 说明 |
|------|------|------|
| 源代码 | `/zhdd/home/hengzhang/code/crysllmgen-main/` | 全部项目代码 |
| 训练模型 | `out/mp_20/04052026/102707/model_final.pt` | Conditional 模型 |
| 训练模型 | `out/mp_20/04052026/120614/model_final.pt` | Cross-attn 模型 |
| 训练模型 | `out/mp_20/06052026/092559/model_final.pt` | Spatial 模型 |
| 采样结果 | `results/llama3_sample_llmonly_mp_20_10000.pt` | LLM-only 采样 |
| 采样结果 | `results/llama3_sample_unconditional_mp_20_10000.pt` | Unconditional 采样 |
| 采样结果 | `results/llama3_sample_cond_mp_20_10000.pt` | Conditional 采样 |
| 采样结果 | `results/llama3_sample_crossattn_mp_20_10000.pt` | Cross-attn 采样 |
| 采样结果 | `results/llama3_sample_spatial_mp_20_10000.pt` | Spatial 采样 |
| 评估日志 | `results/eval_spatial.log` 等 | 各方法评估日志 |
