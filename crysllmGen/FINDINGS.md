# crysllmGen 实验结果与关键发现 (2026-05-07, 最终版)

## 方法论

LLM + 扩散模型晶体结构生成，MP-20 数据集，7 种方法对比。

| 方法 | LLM 角色 | 扩散模型角色 |
|------|---------|-------------|
| LLM-only | 生成全部 | 不使用 |
| Unconditional | 生成初始化 | 精炼 |
| Conditional (additive) | 生成+特征注入 | 精炼 |
| Cross-attn | 生成+特征注入 | 精炼 |
| Spatial cross-attn | 生成+特征注入 | 精炼 |
| **GT comp + rand init** | 不使用 | **从纯噪声生成全部** |
| Multi-scale (lattice-first) | 生成初始化 | 多尺度精炼 |

## 最终评估结果 (全部方法 n≈10000)

| Method | comp_valid | struct_valid | valid | wdist_density | wdist_#elem | cov_recall | cov_precision |
|--------|-----------|-------------|-------|--------------|-----------|------------|---------------|
| **GT comp + rand init** | **91.87** | 99.69 | **91.61** | 0.600 | **0.012** | **99.83** | 98.3 |
| Unconditional | 86.27 | 99.93 | 86.22 | **0.102** | 0.204 | 99.15 | **99.18** |
| Cross-attn | 86.27 | 99.29 | 85.75 | 1.514 | 0.204 | 94.43 | 95.27 |
| LLM-only | 86.27 | 92.08 | 79.92 | 0.976 | 0.222 | 96.75 | 97.78 |
| Conditional (additive) | 86.27 | 99.94 | 86.22 | 3.563 | 0.204 | 53.54 | 95.95 |
| Spatial | 86.27 | 92.94 | 80.38 | 1.125 | 0.216 | 55.87 | 99.41 |
| Multi-scale (lattice-first) | 85.2 | 99.9 | 85.2 | 0.076 | 0.196 | 94.31* | 98.8 |

> *Multi-scale 为 n=1000 评估，需扩大至 n=10000 确认。

## 关键发现

### 1. GT 组分 + 扩散 = 历史最佳 (核心消融实验)

给定 GT 组分 + 纯随机坐标/晶格，无条件扩散达到：
- **cov_recall=99.83%** (历史最高)
- **valid=91.61%** (历史最高)
- **comp_valid=91.87%** (LLM 只有 86.27)
- **wdist_num_elems=0.012** (元素分布近乎完美)

**结论：扩散模型的生成能力远超 LLM。LLM 是当前系统的绝对瓶颈。**

### 2. 无条件扩散是最优 LLM-based 方法

99.15% recall, 所有 LLM 条件化方案 (additive/cross-attn/spatial) 均降低性能。

### 3. 多尺度晶格优先改善了密度分布

wdist_density 从 0.102 降至 0.076 (-25%)，struct_valid 保持 99.9%。

### 4. LLM 擅组分不擅坐标

LLM-only cov_recall=96.75% (组分合理) 但 struct_valid=92.08% (坐标不准)。

## 方法论洞察

```
LLM 擅长: 元素组合 (comp) — 覆盖率 96.75%
LLM 不擅长: 精确坐标 + 组分准确性 — comp_valid 仅 86.27
扩散擅长: 结构精炼 — struct_valid 99.93%, 给定完美组分可达 99.83% recall
扩散上限: comp_valid=91.87, valid=91.61, recall=99.83
```

**最优路径**: 解耦 LLM 和扩散——LLM 只负责组分，扩散从随机噪声生成结构。当前上限高于所有 LLM-based 方法。

## 代码模块

| 模块 | 路径 | 用途 |
|------|------|------|
| CFG | `exp_llm_cfg/` | Classifier-Free Guidance 训练+采样 |
| 多尺度 | `exp_llm_multiscale/` | 晶格优先多尺度去噪 |
| 解耦生成 | `exp_llm_decoupled/` | GT 组分→无条件扩散 |

完整代码在 `/zhdd/home/hengzhang/code/crysllmgen-main/`。
