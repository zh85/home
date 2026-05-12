# 提升 LLM 组分生成能力 — 候选实验方案

## 瓶颈分析

当前 LLaMA-3-8B 的 `comp_valid=86.27%`，扩散模型在 GT 组分下可达 `91.87%`，差距约 5.6pp。

| 指标 | LLM-only | Unconditional | GT comp + diffusion |
|------|----------|---------------|-------------------|
| comp_valid | 86.27 | 86.27 | **91.87** |
| cov_recall | 96.75 | 99.15 | **99.83** |
| valid | 79.92 | 86.22 | **91.61** |

**结论**: LLM 是系统瓶颈，提升 LLM 组分生成质量是最直接方向。

---

## 当前训练状态

| 参数 | 值 |
|------|-----|
| 基座模型 | **Meta-Llama-3-8B** |
| 微调方式 | LoRA (rank=16, alpha=64), 42M 可训练参数 (0.52%) |
| 数据集 | mp_20 (train=27,136, val=9,047) |
| 训练量 | **仅 1 epoch, 848 steps, ~1.4 小时** |
| Train Loss (最终) | **0.566** |
| 采样参数 | temperature=1.0, top_p=0.7 |
| Checkpoint | `exp/test_run/` |

> ⚠️ **关键问题**: 模型仅训练了 1 个 epoch，train loss=0.566 距收敛尚有空间（对比：LLaMA-2-7B 训练 32 epoch 后 loss=0.421）。当前 comp_valid=86.27% 是欠训练状态。

---

## 方案 A: 多 Epoch 充分训练 ⭐ 最优先

### 原理
当前模型只跑了 1 epoch (848 steps)，train loss=0.566。先让模型充分收敛，建立新的 baseline，后续所有方案在此基础上叠加。

### 步骤
1. 用现有 `llm_finetune_llama3.py` 脚本，`--num-epochs 5`（默认），`--run-name llama3-8b-mp-5epoch`
2. 或者从 `exp/test_run/checkpoint-848` 继续训练 `--resume-dir exp/test_run/checkpoint-848 --num-epochs 5`
3. 监控 eval loss 确定是否过拟合
4. 充分训练后重新跑完整 pipeline：采样 → 扩散精炼 → 评估

### 预期
comp_valid 提升 2-4pp（base model 充分训练效果）

### 关键实现细节
- 损失曲线尚在下降，1 epoch 明显不足
- 可先跑 3 epoch 看 loss 是否继续下降
- 采样温度可能需要调整（充分训练后 token 分布更尖锐）

---

## 方案 B: DPO 物理反馈微调

### 原理
充分训练后，用 DPO 对 LLM 做偏好对齐。M3GNet / CHGNet 预测 `energy_above_hull` 作为自动偏好信号。

### 步骤
1. 用充分训练的 LLM 对每个 prompt 采样 8~16 个候选组分
2. CHGNet 计算每个候选的 `e_hull`（越低越稳定）
3. 构造偏好对：`chosen = 低 e_hull`, `rejected = 高 e_hull`
4. DPO 微调 LLM 1~3 个 epoch（LoRA）
5. 重新评估 comp_valid / cov_recall

### 预期
comp_valid 额外提升 2~4pp

### 参考论文
- [PLaID++](https://arxiv.org/abs/2509.07150) (2025.09) — Wyckoff + DPO + MLIP reward → 无条件生成 +115%, S.U.N. +50%
- [Superalloys DPO](https://openreview.net/forum?id=nEF9q1UmEZ) (NeurIPS 2025) — DPO + Thermo-Calc → 开源模型超越 GPT-4.1
- [Crystal GRPO](https://arxiv.org/abs/2511.07158) (2025.11) — GRPO + 多目标 reward

### 关键实现细节
- Reward 函数: `reward = -e_hull` 或 `reward = exp(-e_hull / kT)`
- 偏好对阈值: e_hull 差距 > 0.05 eV/atom 才构成有效 pair
- 每轮 DPO 后提升采样温度对抗 entropy collapse（PLaID++ 策略）

---

## 方案 C: Best-of-N + 物理重排序

### 原理
不修改模型权重。推理时采样 N 个候选，用物理评分函数选最优。

### 步骤
1. 充分训练后的 LLM（或 DPO 后），每个 prompt 采样 N=16/32/64 个结构
2. 评分函数: `score = w1 * charge_balance + w2 * (-e_hull) + w3 * composition_validity`
3. 选 top-1 送入扩散模型精炼
4. 扫描 N 画 comp_valid scaling 曲线

### 预期
N=16 时 comp_valid 提升 2~3pp（零训练成本，可与 A/B 叠加）

### 参考论文
- [FusioN](https://arxiv.org/abs/2510.00931) (Cohere, 2025.10) — LLM judge 合成 N 个候选最优元素
- [matgen_baselines](https://arxiv.org/abs/2501.02144) (Materials Horizons, 2025) — CHGNet 后筛选
- [TreeBoN](https://aclanthology.org/2025.findings-emnlp.1140/) (EMNLP 2025) — 推测树搜索 + BoN

---

## 方案 D: 组分约束解码

### 原理
LLM 生成每个 token 时，mask 掉违反化学规则的 token。

### 步骤
1. 构建化学约束规则集：电荷中性、氧化态合法范围、元素配对规则
2. 自回归生成时每步调用 mask 函数
3. ~10-20% 额外推理开销

### 预期
comp_valid 大幅提升（+5pp），多样性可能下降

### 参考论文
- [SmiSelf](https://aclanthology.org/2025.emnlp-main.1350/) (EMNLP 2025) — 100% 分子有效性
- [VALID-Mol](https://arxiv.org/abs/2506.23339) (2025.06) — 3% → 83% 有效结构

---

## 方案 E: Wyckoff 表示 + 对称性注入

### 原理
替换 CIF 文本表示，改用 Wyckoff 位置编码，显式注入空间群对称性。

### 预期
struct_valid 显著提升，token 数减少 ~14%

### 参考论文
- [PLaID++](https://arxiv.org/abs/2509.07150) — Wyckoff 编码: 185.5 vs 214.7 tokens (-14%)
- [AlchemBERT](https://www.sciencedirect.com/science/article/pii/S2666386425003236) (2025) — NL 晶体描述 MAE -40.3%
- [CrystalICL](https://aclanthology.org/2025.emnlp-main.929/) (EMNLP 2025)

---

## 优先级与路线图

| 优先级 | 方案 | 训练成本 | 预期收益 | 时间 |
|--------|------|---------|---------|------|
| **P0** | A: 多 Epoch 充分训练 | 1× GPU, ~7h | +2~4pp comp_valid | 1-2 天 |
| **P1** | B: DPO 物理反馈 | 1× GPU, ~1天 | +2~4pp (叠加 A) | 1 周 |
| **P1** | C: Best-of-N 重排序 | 零 | +2~3pp (叠加 A+B) | 2-3 天 |
| **P2** | D: 约束解码 | 零 | +5pp (多样性↓) | 3-5 天 |
| **P2** | E: Wyckoff 表示 | 1× GPU + 数据工程 | 长期收益 | 2-3 周 |

### 建议执行顺序
1. **先跑 A** — 充分训练是前提。当前 1 epoch / loss=0.566 明显不足
2. **A 完成后跑 B** — DPO 是目前领域最热、验证最多的方法
3. **C 随时可跑** — 零训练成本，对 A/B 结果叠加测试
4. **D/E 作为备选** — 工程量大，看 A+B 效果决定

### 评估指标
- 主指标: comp_valid, cov_recall
- 辅助指标: struct_valid, wdist_density, wdist_#elem
- 稳定性: energy_above_hull (CHGNet) 均值/中位数

---

## 相关综述与基准

- [Awesome AI for Materials Generation](https://github.com/zhixunlee/awesome-ai-for-materials-generation) — 2025 综述
- [LeMat-GenBench](https://neurips.cc/virtual/2025/loc/san-diego/128979) (NeurIPS 2025) — 12 模型统一基准
- [matgen_baselines](https://github.com/Bartel-Group/matgen_baselines) — 生成模型 vs 传统方法
