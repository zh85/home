# 提升 LLM 组分生成能力 — 候选实验方案

## 瓶颈分析

当前 LLM (LLaMA-2-7B) 的 `comp_valid=86.27%`，而扩散模型在 GT 组分下可达 `91.87%`，差距约 5.6pp。

| 指标 | LLM-only | Unconditional | GT comp + diffusion |
|------|----------|---------------|-------------------|
| comp_valid | 86.27 | 86.27 | **91.87** |
| cov_recall | 96.75 | 99.15 | **99.83** |
| valid | 79.92 | 86.22 | **91.61** |

**结论**: LLM 是系统绝对瓶颈，提升 LLM 组分生成质量是最直接的优化方向。

---

## 方案 A: DPO 物理反馈微调 ⭐ 推荐

### 原理
对当前 fine-tuned LLaMA-2-7B 做 DPO（Direct Preference Optimization），用 M3GNet / CHGNet 预测的 `energy_above_hull` 作为偏好信号。物理模拟器自动打分，无需人工标注。

### 步骤
1. 用当前 LLM 对每个 prompt 采样 8~16 个候选组分
2. 用 CHGNet 计算每个候选的 `e_hull`（越低越稳定）
3. 构造偏好对：`chosen = 低 e_hull 候选`, `rejected = 高 e_hull 候选`
4. DPO 微调 LLM 1~3 个 epoch（LoRA 可选项）
5. 重新评估 comp_valid / cov_recall

### 预期
comp_valid 提升 2~4pp

### 参考论文
- [PLaID++: Preference Aligned Language Model for Targeted Inorganic Materials Design](https://arxiv.org/abs/2509.07150) (2025.09)
  - Wyckoff 表示 + DPO + MLIP reward → 无条件生成 +115%，S.U.N. +50%
- [Preference Learning from Physics-Based Feedback (Superalloys)](https://openreview.net/forum?id=nEF9q1UmEZ) (NeurIPS 2025)
  - DPO + Thermo-Calc → 开源模型超越 GPT-4.1
- [Guiding Generative Models via GRPO for Crystals](https://arxiv.org/abs/2511.07158) (2025.11)
  - GRPO + 多目标 reward（稳定性+多样性+新颖性）

### 关键实现细节
- Reward 函数：`reward = -e_hull` 或 `reward = exp(-e_hull / kT)`
- 偏好构造阈值：e_hull 差距 > 0.05 eV/atom 才构成有效 pair
- 温度升调度：每轮 DPO 后提升采样温度对抗 entropy collapse（PLaID++ 策略）

---

## 方案 B: Best-of-N + 物理重排序 ⭐ 推荐

### 原理
不改模型权重。推理时采样 N 个候选，用物理评分函数选最优，送入扩散模型精炼。

### 步骤
1. 对每个 prompt 采样 N=16/32/64 个结构
2. 评分函数：`score = w1 * charge_balance + w2 * (-e_hull) + w3 * composition_validity`
3. 选 top-1 送入扩散模型
4. 扫描 N 画 comp_valid 随 N 的 scaling 曲线

### 预期
N=16 时 comp_valid 提升 2~3pp（零训练成本）

### 参考论文
- [FusioN: Making, not Taking, the Best of N](https://arxiv.org/abs/2510.00931) (Cohere, 2025.10)
  - LLM judge 合成 N 个候选最优元素，一致优于 vanilla BoN
- [matgen_baselines](https://arxiv.org/abs/2501.02144) (Materials Horizons, 2025)
  - CHGNet 后生成筛选，实质提升成功率
- [TreeBoN](https://aclanthology.org/2025.findings-emnlp.1140/) (EMNLP 2025)
  - 推测树搜索 + BoN，DPO token 级 reward 指导扩展/剪枝

### 关键实现细节
- 物理评分函数可用 CHGNet（速度快）或 M3GNet（更准）
- 评分公式建议：`score = -e_hull - 10 * |charge|`（电荷中性硬约束）
- 可叠加方案 D（约束解码）进一步减少无效候选

---

## 方案 C: 升级 LLM 基座模型

### 原理
LLaMA-2-7B → LLaMA-3.1-8B 或 Qwen-2.5-7B，利用更强的基础能力和更好的训练数据。

### 步骤
1. 用当前 mp_20 数据重新 fine-tune LLaMA-3.1-8B（或 Qwen-2.5-7B）
2. 保持相同的训练配置和扩散模型
3. 对比 comp_valid / cov_recall

### 预期
comp_valid 提升 1~3pp

### 参考论文
- [CrysText](https://svivek.com/research/mohanty2025crystext.html) (2025) — LLaMA-3.1-8B + QLoRA → CIF 生成
- [PLaID++](https://arxiv.org/abs/2509.07150) — Qwen-2.5-7B
- [CrystalLLM](https://www.nature.com/articles/s41467-024-52439-5) (Nat. Comm. 2024) — 专用 GPT-2 200M，DFT 验证 20 个稳定新材料

### 关键实现细节
- Qwen-2.5-7B 对中文/结构化文本的 tokenizer 效率更优
- LLaMA-3.1-8B 的 RoPE 扩展上下文可能有助于长晶体序列
- 不要替换扩散模型，只替换 LLM 部分

---

## 方案 D: 组分约束解码

### 原理
LLM 生成每个 token 时，mask 掉违反化学规则的 token（电荷不平衡、氧化态冲突等）。

### 步骤
1. 构建化学约束规则集：电荷中性、氧化态合法范围、元素配对规则（如某些元素不能共存）
2. 规则集编码为 token mask 生成函数
3. 在 LLM 自回归生成时，每步调用 mask 函数过滤违规 token
4. 约 10~20% 额外推理开销

### 预期
comp_valid 大幅提升（+5pp 以上），但多样性可能下降

### 参考论文
- [SmiSelf](https://aclanthology.org/2025.emnlp-main.1350/) (EMNLP 2025) — 后处理矫正实现 100% 分子有效性
- [VALID-Mol](https://arxiv.org/abs/2506.23339) (2025.06) — 验证管线：3% → 83% 有效结构
- [TSMMG](https://www.scilit.com/publications/ca324af599111069f675e7d519a3ef00) (BMC Biology, 2025) — Teacher-student 多约束分子生成，>99% 有效性

### 关键实现细节
- 惰性约束：只 mask 确定性违规（如电荷符号错误），不 mask 可能性违规
- 可与方案 B 叠加：约束解码减少无效候选 → BoN 更高效

---

## 方案 E: Wyckoff 表示 + 对称性注入

### 原理
替换当前 CIF 文本表示，改用 Wyckoff 位置编码，显式注入空间群对称性信息。让 LLM 天然学习物理约束。

### 步骤
1. 改造数据预处理：CIF → Wyckoff 文本表示
2. 重新 fine-tune LLM（或对现有 LLM 做轻量适配）
3. 对比 comp_valid / struct_valid / token 效率

### 预期
struct_valid 显著提升（+3~5pp），comp_valid 间接受益，token 数减少 ~14%

### 参考论文
- [PLaID++](https://arxiv.org/abs/2509.07150) — Wyckoff 编码：185.5 vs 214.7 tokens/crystal (-14%)
- [AlchemBERT](https://www.sciencedirect.com/science/article/pii/S2666386425003236) (2025) — NL 晶体描述 MAE 降低 40.3%
- [CrystalICL](https://aclanthology.org/2025.emnlp-main.929/) (EMNLP 2025) — 空间群感知 tokenization + few-shot 生成

### 关键实现细节
- Wyckoff 表示格式示例：`Space group: Pnma; Wyckoff: Ba 4c (x,y,0.25) Fe 4a (0,0,0) O 8d (x,y,z)`
- 可用 pymatgen 的 `SpaceGroupAnalyzer` 自动转换
- 需要清理掉不满足 Wyckoff multiplicity 规则的结构

---

## 优先级与路线图

| 优先级 | 方案 | 训练成本 | 预期收益 | 时间 |
|--------|------|---------|---------|------|
| **P0** | B: Best-of-N 重排序 | 零 | +2~3pp comp_valid | 2-3 天 |
| **P0** | A: DPO 物理反馈 | 1× GPU, ~1天 | +2~4pp comp_valid | 1 周 |
| **P1** | C: 升级基座模型 | 1× GPU, ~1天 | +1~3pp comp_valid | 1 周 |
| **P1** | D: 约束解码 | 零 | +5pp comp_valid (多样性↓) | 3-5 天 |
| **P2** | E: Wyckoff 表示 | 1× GPU + 数据工程 | 长期收益 | 2-3 周 |

### 建议执行顺序
1. **先跑 B** — 快速探明 LLM 采样质量的上界（不改模型，N 够大能到多好？）
2. **并行跑 A** — DPO 是 2025 最验证过的方向，PLaID++ 和 Superalloys DPO 都证实有效
3. **C 作为备选** — 如果 A+B 效果不达预期，升级基座是保底方案
4. **D 可与 B 叠加** — 约束+重排序联合使用
5. **E 作为长期投入** — 如果方向确定做晶体表示优化

### 评估指标
- 主指标：comp_valid, cov_recall
- 辅助指标：struct_valid, wdist_density, wdist_#elem
- 稳定性指标：energy_above_hull (CHGNet) 均值和中位数

---

## 相关综述与基准

- [Awesome AI for Materials Generation](https://github.com/zhixunlee/awesome-ai-for-materials-generation) — 2025 综述，覆盖所有生成方法
- [LeMat-GenBench](https://neurips.cc/virtual/2025/loc/san-diego/128979) (NeurIPS 2025) — 12 模型统一基准，HuggingFace leaderboard
- [matgen_baselines](https://github.com/Bartel-Group/matgen_baselines) — 生成模型 vs 传统方法系统对比
