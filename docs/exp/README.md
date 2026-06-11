# NLA 实验文档

**模型**: Qwen2.5-7B-Instruct, layer 20, NLA AV/AR
**实验周期**: 2026-05-20 ~ 2026-05-23
**扫描规模**: ~4000 激活向量, 50+ 实验条件

---

## 目录结构

```
docs/exp/
├── README.md                          # 本文件
├── index.md                           # 完整阅读指南与文件清单
│
├── architecture/                      # 前置知识
│   └── 技术架构与实验流程详解.md        # AV/AR 系统架构、注入原理、数据流
│
├── completed/                         # 已完成旧实验归档
│   ├── README.md
│   ├── round1_baseline/               # Round 1: 定性/定量基线
│   ├── critique/                      # R1 后批判与方向转折
│   ├── round2_position/               # Round 2: 隐藏状态 + 位置分辨
│   ├── round3_original_cases/         # Round 3: 原文 case 近似复现
│   ├── team_reports/                  # 团队汇报版本
│   ├── configs/                       # 已完成实验配置与 runbook
│   ├── latex_report/                  # 旧 LaTeX 综合报告及构建产物
│   └── _obsolete/                     # 被替代的旧实验材料
│
├── original_paper_evals/              # 原文后半段正式评估工作区
│   ├── README.md                      # 命名空间说明与文件命名规则
│   ├── plans/                         # 后续实验计划
│   │   ├── no_training.md
│   │   └── training_related.md
│   ├── shared/                        # 共享 schema / 指标规范
│   ├── templates/                     # 报告与 run manifest 模板
│   ├── no_training/                   # 现有 AV/AR 可直接跑的实验产物
│   └── training_related/              # 依赖 checkpoint/SAE/oracle/target organism 的实验产物
```

---

## 快速入门

### 阅读路径

- **快速理解** (~1h): `architecture/` → `completed/critique/CRITIQUE.md` → `completed/round2_position/SYNTHESIS_REPORT_R2.md` → `completed/round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md`
- **完整学术** (~3h): 按 `index.md` 中的完整学术路径
- **只看结论** (~20min): 见 `index.md` 中的快速路径
- **规划后续实验**: 先读 `original_paper_evals/README.md`，再读 `original_paper_evals/plans/no_training.md` 和 `original_paper_evals/plans/training_related.md`

### 原始数据

所有 `.parquet` 激活向量数据、`.csv` 评分数据和 `.npy` 重建向量均已包含在各轮次的 `data/` 子目录中。

### 复现实验

```bash
conda activate nla
source scripts/qwen_nla_env.sh

# 生成parquet (需要GPU)
python scripts/exp_generate_parquets.py --config docs/exp/completed/configs/plan_a.yaml

# 启动SGLang推理
CUDA_VISIBLE_DEVICES=4 PORT=30000 bash scripts/launch_qwen_nla_sglang.sh

# 运行推理
bash scripts/exp_run_inference.sh plan_a
```

详见 `completed/configs/original_cases_l20_RUNBOOK.md` 和 `../docs/setup.md`。

---

## 关键指标速查

| 指标 | 含义 | 方向 |
|---|---|---|
| **cos** | NLA解码文本与参考嵌入的余弦相似度 | 越高越好 (~0.95上限) |
| **MSE** | AR Critic方向MSE (缩放) | 越低越好 |
| **norm** | 残差流激活L2范数 | 低值可能=表征塌缩 |
| **FVU** | AR重建的未解释方差比例 | 越低越好 |
| **delta LM loss** | 替换激活后next-token CE增量 | 越低越好 |

## 参考来源

- Anthropic Blog: https://www.anthropic.com/research/natural-language-autoencoders
- Paper: https://transformer-circuits.pub/2026/nla/index.html
- Local PDF: `../Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.pdf`
