# 化工智能Agent — 学术论文综述 (2024–2026)

## 一、多智能体 (Multi-Agent) 化工优化

### LLM-guided Chemical Process Optimization with a Multi-Agent Approach
- **作者**: Tong Zeng, Srivathsan Badrinarayanan 等 (CMU)
- **发表**: *Machine Learning: Science and Technology*, 2025
- **链接**: https://iopscience.iop.org/article/10.1088/2632-2153/ae2382
- **要点**: 基于 AutoGen 多Agent框架 + OpenAI o3, 自主推断操作约束并协作优化 HDA 化工过程, 比网格搜索提速 **31倍**, 20分钟内收敛

### CeProAgents: A Hierarchical Agents System for Automated Chemical Process Development
- **发表**: 2025
- **要点**: 层次化多Agent系统, 分工协作完成化工过程开发六类任务

### MOSES: Multi-agent Ontology System for Explainable Knowledge Synthesis
- **发表**: ChemRxiv, 2025年10月
- **链接**: https://chemrxiv.org/engage/chemrxiv/article-details/68d928473e708a7649a8db7a
- **要点**: 自动化本体构建 + 多Agent推理, 在复杂化学问题上超越 GPT-4.1 和 o3

## 二、催化与材料设计 Agent

### CATDA: Distilling Knowledge from Catalysis Literature with Long-Context LLM Agents
- **作者**: Honghao Chen, Xiaonan Wang 等 (清华大学化工系)
- **发表**: *ACS Catalysis*, 2025
- **链接**: https://pubs.acs.org/doi/10.1021/acscatal.5c06431
- **要点**: 长上下文LLM Agent从催化文献中自动提取知识图谱 (F1=**0.983**), 速度比人工快 **12倍**

### Conversational LLM AI Agent for Accelerated Synthesis of MOFs Catalysts (MOFsyn Agent)
- **作者**: Jing Lin 等
- **发表**: *ACS Nano*, 2025
- **链接**: https://pubmed.ncbi.nlm.nih.gov/40551463/
- **要点**: RAG增强Agent优化MOF合成, Ni@UiO-66(Ce) 烯烃加氢 **100%** 转化率和选择性

### Chemistry Foundation Models for Multi-Agent Workflows in Catalyst and Materials Design
- **作者**: Jie Ren 等 (IBM Research)
- **发表**: NeurIPS 2025
- **链接**: https://research.ibm.com/publications/leveraging-chemistry-foundation-models-to-facilitate-structure-focused-retrieval-augmented-generation-in-multi-agent-workflows-for-catalyst-and-materials-design
- **要点**: 化学基础模型 + 多模态模型 + RAG 多Agent工作流

## 三、化工流程自动化与仿真

### AutoChemSchematic AI: Agentic Physics-Aware Automation for Chemical Manufacturing Scale-Up
- **发表**: NeurIPS 2025
- **要点**: 闭环物理感知框架, 集成领域SLM + 层次知识图谱 + 化工模拟器, 自动生成工业级 PFD/PID

### The Potential and Challenges of LLM Agent Systems in Chemical Process Simulation
- **作者**: 杜文莉、杨绍毅 (华东理工大学)
- **发表**: *Frontiers of Chemical Science and Engineering*, 2025
- **要点**: 综述, 提出「多模态任务感知—自主规划—知识驱动迭代优化」三支柱框架

### Sketch2Simulation: Automating Flowsheet Generation via Multi-Agent LLMs
- **发表**: 2025
- **要点**: 端到端多Agent系统, 将PFD直接转换为可执行 Aspen HYSYS 模拟文件

## 四、行业应用 Agent

### IM-Chat: Multi-agent LLM Framework for Knowledge Transfer in Injection Molding
- **发表**: *Journal of Manufacturing Systems*, 2025
- **链接**: https://www.sciencedirect.com/science/article/abs/pii/S0278612525002687
- **要点**: 注塑成型多Agent框架, RAG文档检索 + 扩散模型过程优化, 160个基准任务高准确率

## 趋势总结

| 方向 | 说明 |
|------|------|
| Multi-Agent 协作 | AutoGen/GroupChat 架构, 分工-协商-集成 |
| RAG + 知识图谱 | 文献提取 → 知识图谱 → 推理决策 |
| 长上下文推理 | 跨文档关联合成-结构-性能关系 |
| 物理仿真闭环 | Agent ↔ Aspen/IDAES 模拟器联动 |
| 可解释性 | 黑箱 → 可追溯推理路径 (MOSES) |
