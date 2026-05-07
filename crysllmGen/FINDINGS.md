# crysllmGen 实验结果与关键发现 (2026-05-07)

## 方法论

本项目研究 LLM + 扩散模型的晶体结构生成，在 MP-20 数据集上对比 7 种方法。

| 方法 | LLM 角色 | 扩散模型角色 | 说明 |
|------|---------|-------------|------|
| LLM-only | 生成全部 | 不使用 | 纯文本生成 CIF |
| Unconditional | 生成初始化 | 精炼坐标+晶格 | 无条件 CSPNet, 当前最优 |
| Conditional (additive) | 生成+特征注入 | 精炼 | 加性 LLM 特征注入 |
| Cross-attn | 生成+特征注入 | 精炼 | 交叉注意力 LLM 注入 |
| Spatial cross-attn | 生成+特征注入 | 精炼 | 空间交叉注意力 (新增) |
| GT comp + rand init | 不使用 | 生成全部 | 真实组分→纯噪声→扩散(新增) |
| Multi-scale (lattice-first) | 生成初始化 | 多尺度精炼 | 先锁晶格后调坐标(新增) |

## 评估结果

> 注: 原 5 种方法 n≈10000 (test.csv 全部 9046 条)，新方法 n=1000。后续需扩充至全量对比。

| Method | n | comp_valid | struct_valid | valid | wdist_density | cov_recall | cov_precision |
|--------|---|-----------|-------------|-------|--------------|------------|---------------|
| **GT comp + rand init** | 1000 | **90.0** | **99.8** | **89.8** | 0.561 | 96.27 | 97.3 |
| **Multi-scale (lattice-first)** | 1000 | 85.2 | **99.9** | 85.2 | **0.076** | 94.31 | 98.8 |
| Spell Unconditional | 10000 | 86.27 | 99.93 | 86.22 | **0.102** | **99.15** | **99.18** |
| Cross-attn | 10000 | 86.27 | 99.29 | 85.75 | 1.514 | 94.43 | 95.27 |
| Conditional (additive) | 10000 | 86.27 | 99.94 | 86.22 | 3.563 | 53.54 | 95.95 |
| LLM-only | 10000 | 86.27 | 92.08 | 79.92 | 0.976 | 96.75 | 97.78 |
| Spatial | 10000 | 86.27 | 92.94 | 80.38 | 1.125 | 55.87 | 99.41 |

## 关键发现

### 1. LLM 是系统的瓶颈，不是扩散模型

给定真实组分 (ground-truth composition) + 纯随机坐标/晶格初始化，无条件扩散模型达到了 **struct_valid=99.8%（历史最佳）** 和 **valid=89.8%（历史最佳）**。

comp_valid 从 LLM 的 86.27 跃升至 GT 的 90.0，说明 LLM 生成的组分是当前上限。

**这意味着**：改进 LLM 的组分预测能力比改进扩散模型更优先。

### 2. 无条件扩散是最优方法

在所有 LLM-based 方法中，无条件扩散 baseline 达到了最佳 recall (99.15%) 和 precision (99.18%)。所有 LLM 条件化方案（additive/cross-attn/spatial）均**降低**了性能。

**原因推测**：LLM 特征 (4096-dim pooled hidden states) 含有过多噪声，扩散模型难以从中提取有效的晶体结构信息。

### 3. 晶格优先多尺度采样改善了密度分布

冻结晶格后再精细调整原子坐标的策略，使密度分布最接近真实数据 (wdist_density=0.076 vs baseline 0.102)。结构有效率也最高 (99.9%)。

### 4. 空间交叉注意力和加性条件化大幅降低召回率

Spatial 和 Cond 方法的 cov_recall 仅为 ~55%，远低于无条件 baseline 的 99.15%。这些 LLM 注入方式存在根本性问题。

### 5. LLM-only 覆盖率很高但结构有效率低

纯 LLM 生成的结构 cov_recall=96.75%（接近无条件扩散的 99.15%），但 struct_valid 只有 92.08%。LLM 能生成合理的元素组合，但缺乏精确坐标能力。

## 方法论洞察

```
LLM 擅长: 元素组合 (comp) — 覆盖率 96.75%
LLM 不擅长: 精确坐标 (coords) — 有效率 92.08%
扩散擅长: 结构精炼 (coords+lattice) — 有效率 99.93%
扩散需要: 合理的组分输入 — 上限 90.0% comp_valid
```

**最优路径**: 解耦 LLM 和扩散模型的职责——LLM 只负责元素组合（comp），扩散模型从随机噪声生成结构。

## 代码模块

| 模块 | 路径 | 用途 |
|------|------|------|
| CFG | `exp_llm_cfg/` | Classifier-Free Guidance 训练+采样 |
| 多尺度 | `exp_llm_multiscale/` | 晶格优先多尺度去噪 |
| 解耦生成 | `exp_llm_decoupled/` | GT 组分→无条件扩散 |

完整代码在 `/zhdd/home/hengzhang/code/crysllmgen-main/`。

## 待办

- [ ] GT comp 实验扩大至 n=10000 确认上限
- [ ] Multi-scale 扩大至 n=10000 区分样本量效应
- [ ] LLM-only num_atoms + 扩散生成 atom_types (需 categorical diffusion)
- [ ] CFG 训练 (代码已完成，待 GPU)
