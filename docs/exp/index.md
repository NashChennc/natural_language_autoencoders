# NLA 实验报告索引与阅读指南

**生成日期**: 2026-06-03
**实验总周期**: 2026-05-20 ~ 2026-05-23
**实验者**: nashchennc
**模型**: Qwen2.5-7B-Instruct, layer 20, NLA AV/AR

---

## 整体图景

整个实验历时约 **3 天**，分 **3 轮**，围绕 Qwen2.5-7B-Instruct 第 20 层的 NLA AV/AR 系统，从定性基线逐步深入到 Anthropic 原文 Case 复现。共扫描了 **~4000 个激活向量**，横跨 50+ 个实验条件。

```
                    ┌──────────────────────────────────────┐
                    │         📄 前置知识（必读）             │
                    │ 技术架构与实验流程详解.md (36KB)        │
                    │ qwen_nla_reproduction_detailed (11KB)  │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │         📄 Round 1 地基               │
                    │ 实验报告_R1.md (44KB)                  │
                    │ SYNTHESIS_REPORT.md (38KB)             │
                    │ quant_report.md (3.4KB)                │
                    │ CRITIQUE.md (25KB)  ← 对 R1 的批判     │
                    │ NEXT_STEPS.md (10KB) ← 由批判驱动      │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼────────┐  ┌────────▼────────┐  ┌───────▼──────────┐
    │ 📄 Round 2 深化   │  │ 📄 Position 专题 │  │ 📄 团队报告视角   │
    │ 实验报告_R2 (47KB)│  │ position_report  │  │ NLA_Qwen_批判式   │
    │ SYNTHESIS_R2(48KB)│  │ (1.3KB)          │  │ 复现实验报告(7.4KB)│
    └─────────┬────────┘  └────────┬────────┘  └───────┬──────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │      📄 Round 3 原文 Case 复现         │
                    │ ORIGINAL_CASES_L20_汇总报告.md (45KB)  │
                    │ gemmascope_reconstruction_* (各汇总)   │
                    │ scores_by_condition.md (12KB)          │
                    └──────────────────────────────────────┘
```

---

## 逐文件说明与阅读顺序

### 第零层：前置知识（先读，建立概念框架）

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **0.1** | `architecture/技术架构与实验流程详解.md` | AV/AR 双模型系统架构、注入数学原理、sidecar 契约、完整数据流（datagen → inference → scoring）、SGLang 推理管线 | ⭐⭐⭐⭐⭐ |
| **0.2** | `../../docs/qwen_nla_reproduction_detailed_2026-05-20.md` | 环境搭建记录：模型路径、conda 环境、GPU 配置、FlashInfer 后端恢复；是复现的"操作手册" | ⭐⭐⭐ |

**为什么先读**：没有这两个文件，你无法理解任何实验报告中的 cos/MSE/norm/L0 proxy/FVU/delta LM loss 是什么，也无法理解"注入标记 ㈎"和"injection_scale=150"的含义。

---

### 第一层：Round 1 — 定性基线（20 条件 × 16 向量 = 320 向量）

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **1.1** | `completed/round1_baseline/实验报告_R1.md` | 实验设计（Plan A/B/C/D）、全 20 条件定量表、逐 Plan 定性解码分析、关键发现 | ⭐⭐⭐⭐ |
| **1.2** | `completed/round1_baseline/SYNTHESIS_REPORT.md` | R1 的英文综合报告 — 内容与 1.1 重叠但更精炼，Executive Summary 值得一读 | ⭐⭐⭐ |
| **1.3** | `completed/round1_baseline/quant_report.md` | 纯定量数据：391 个向量的 AR Critic cos/MSE/norm 统计表，Plan C 安全梯度呈现 | ⭐⭐⭐⭐ |

**R1 核心发现**：

- **安全梯度**：C1 良性(cos=0.946) → C2 拒绝(0.878) → C5 越狱(0.837) 单调递减
- **范数抑制**：C2 拒绝回答的 norm 只有 C1 的 89.9%
- **C5 位置 1 异常**：norm=81.4，全数据集最低点
- **Plan B 收敛**：5 种系统提示中有 4 种收敛到相同的 AI 身份声明模板

**依赖关系**：1.1 → 1.2（英文版是中文版的精炼），1.3 是 1.1 的定量数据附录。

---

### 第二层：批判与转向

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **2.1** | `completed/critique/CRITIQUE.md` | 对 R1 的致命批评：**(a) 单次生成无法区分条件间差异和随机变异；(b) 全是定性阅读没有量化指标；(c) temperature=0 掩盖解码不确定性；(d) 只取了前 16 个 reply token** | ⭐⭐⭐⭐⭐ |
| **2.2** | `completed/critique/NEXT_STEPS.md` | 由 Critique 驱动的下一步决策：**优先做位置分辨实验（Position-Resolved）而非多样本鲁棒性检查**。解释了为什么 C2 拒绝需要 8+ token 才能达到 C1 在第 1-2 token 的重建保真度——提出了"表征结晶化"假说 | ⭐⭐⭐⭐⭐ |

**为什么这一层极其重要**：CRITIQUE 直接改变了实验方向。R1 报告声称"NLA 可以区分任务类型"被批评为 trivial（训练目标本身就是这个），而"安全梯度"和"表征结晶化"被识别为真正有价值的方向。NEXT_STEPS 是 R1→R2 的桥梁。

**依赖关系**：CRITIQUE 直接引用 `实验报告_R1` 和 `SYNTHESIS_REPORT`；NEXT_STEPS 是对 CRITIQUE 的回应。

---

### 第三层：Round 2 — 隐藏状态检测 + 位置分辨

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **3.1** | `completed/round2_position/实验报告_R2.md` | Plan E（隐藏状态检测 5 变体）+ Position-Resolved（4 条件 × 3 窗口），核心假设 H1-H5 | ⭐⭐⭐⭐⭐ |
| **3.2** | `completed/round2_position/SYNTHESIS_REPORT_R2.md` | R2 英文综合报告——更完整的分析 + 综合讨论（两种表征破坏类型、"提前发现"证据链、late variance 作为安全签名） | ⭐⭐⭐⭐⭐ |
| **3.3** | `completed/round2_position/position_report.md` | C1/C2/C3/C5 的 Early/Mid/Late 三个窗口的纯定量数据（190 向量） | ⭐⭐⭐ |

**R2 核心发现**：

- **E2 抑制错误（cos=0.838）**：与越狱（C5 cos=0.837）统计上不可区分——模型"知道"自己在说假话，即使输出看起来正常
- **E4 提前拒绝信号**：在 token position 0 就检测到拒绝意图——"模型在想拒绝、还没说出口"
- **C2 拒绝的倒 U 形曲线**（0.906 → 0.926 → 0.888）：拒绝表征在中期改善后晚期崩溃，而非预测的单调上升
- **Late variance 分叉**：安全内容 σ<0.04，不安全内容 σ>0.11——这是一个可量化的诊断签名

**依赖关系**：3.1 和 3.2 共享数据但分析深度不同——3.1 偏定量描述，3.2 有更深的综合分析和理论构建。3.3 是原始数据附录。

---

### 第四层：团队报告视角

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **4.1** | `completed/team_reports/` 下的各分报告 | 按主题拆分的团队汇报版本：00 批判、01 定量基线、02 局限性、03 隐藏状态检测、04 安全动态位置分辨 | ⭐⭐ |

这些是 R1+R2 的"对外汇报版本"，内容与前面的报告有重叠但组织方式不同。适合快速了解结论而非细节。

**依赖关系**：完全依赖 R1 和 R2 的数据，是它们的"精炼版"。

---

### 第五层：Round 3 — Original Cases L20 原文复现（最完整、最终版）

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **5.1** | `completed/round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md` | **6 组实验（A-F）× 27 条件 × 3315 向量**的完整复现：Evaluation Awareness、Hidden Motivation、Language Switching、Answer Thrashing、Confabulation、Sampling Strategy。使用 GemmaScope 风格的 FVU/delta LM loss 指标 | ⭐⭐⭐⭐⭐ |
| **5.2** | `completed/round3_original_cases/gemmascope_reconstruction_original_cases_l20_report.md` | GemmaScope 指标的详细方法论和整体统计（3315 行的 FVU/delta LM loss 分布） | ⭐⭐⭐⭐ |
| **5.3** | `completed/round3_original_cases/scores_by_condition.md` | 27 条件 × 6 采样策略的完整评分矩阵 | ⭐⭐⭐ |

**R3 核心发现**（6 组实验）：

| 组 | 主题 | Rows | cos mean | FVU | 关键解读 |
|---|---|---|---|---|---|
| A | Evaluation Awareness | 664 | 0.910 | 0.281 | evaluation 场景有信号，但 benchmark 格式本身低保真 |
| B | Hidden Motivation | 655 | 0.922 | 0.259 | hidden-policy/refusal 比 hidden-preference 更明显 |
| C | Language Switching | 581 | 0.931 | 0.238 | 语言相关 case 整体保真度最高 |
| D | Answer Thrashing | 515 | 0.896 | 0.376 | 最容易触发低保真/冲突状态 |
| E | Confabulation | 541 | 0.907 | 0.296 | specificity pressure 有大量可标注 claims |
| F | Sampling Strategy | 359 | 0.901 | 0.312 | `prompt_final` 是最脆弱的采样窗口 |

**采样策略对比**：

| Strategy | Rows | cos mean | FVU | delta LM loss | 设计意图 |
|---|---|---|---|---|---|
| `prompt_final` | 27 | 0.896 | 0.657 | 0.849 | 测"输出前状态"（最脆弱窗口） |
| `reply_early` | 418 | 0.914 | 0.275 | 0.404 | 刚开始回答时的状态 |
| `reply_mid` | 418 | 0.911 | 0.268 | 0.459 | 回答中段的稳定主题 |
| `reply_late` | 418 | 0.905 | 0.283 | 0.453 | 最终总结/答案附近状态 |
| `event_window` | 848 | 0.916 | 0.268 | 0.351 | 聚焦关键词首次出现位置 |
| `uniform_50` | 1186 | 0.912 | 0.265 | 0.446 | 低假设覆盖 baseline |

**最低保真条件**：

| Condition | cos | FVU | delta LM loss | 说明 |
|---|---|---|---|---|
| D4_final_only | 0.830 | 0.898 | 1.763 | 强输出压缩，base 直接答错 |
| D3_force_wrong_style | 0.839 | 0.694 | 0.075 | 要求"故意有错但不指出" |
| A2_benchmark_format | 0.854 | 0.469 | 1.503 | benchmark + final-only/code 输出 |

**最高保真条件**：

| Condition | cos | FVU | delta LM loss | 说明 |
|---|---|---|---|---|
| C2_chinese_only | 0.952 | 0.190 | 0.478 | 中文任务，稳定高保真 |
| C1_english_only | 0.944 | 0.223 | 0.356 | 英文任务，稳定高保真 |
| C4_implicit_language | 0.941 | 0.233 | 0.268 | 杭州/台北文化 cue 下仍高保真 |

**依赖关系**：这是对 Anthropic NLA 论文原文 case 的直接复现尝试，使用前两轮积累的全部基础设施。所有前面的报告都在为这一层铺路。

---

### 第六层：后续正式评估计划

| 顺序 | 文件 | 内容 | 重要性 |
|:---:|---|---|:---:|
| **6.1** | `original_paper_evals/README.md` | 原文后半段正式评估的统一命名空间、文件结构、命名规则、模板入口 | ⭐⭐⭐⭐⭐ |
| **6.2** | `original_paper_evals/plans/no_training.md` | 无训练复现路线：known-truth eval、steganography、confabulation claim ablation、evaluation awareness、NLA+grader QA、toy auditing workflow | ⭐⭐⭐⭐ |
| **6.3** | `original_paper_evals/plans/training_related.md` | 训练相关复现路线：training curves、多 checkpoint 质量变化、SAE agreement、direct activation oracle、trained hidden-objective auditing benchmark | ⭐⭐⭐⭐ |

**依赖关系**：这两份是计划，不是已完成结果。无训练线只依赖现有 released Qwen L20 AV/AR；训练相关线需要 checkpoint、SAE、oracle 或 target organism。

---

## 推荐的阅读路径

### 快速理解路径（~1 小时）

```
architecture/技术架构与实验流程详解.md
    ↓
completed/critique/CRITIQUE.md          ← 理解实验设计的核心问题
    ↓
completed/round2_position/SYNTHESIS_REPORT_R2.md  ← R1 和 R2 的最佳总结
    ↓
completed/round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md §1-3  ← 最终实验的量化结果
```

### 完整学术路径（~3 小时）

```
0. architecture/技术架构与实验流程详解.md
1. completed/round1_baseline/实验报告_R1.md → quant_report.md
2. completed/critique/CRITIQUE.md → NEXT_STEPS.md         ← 关键转折点
3. completed/round2_position/实验报告_R2.md → SYNTHESIS_REPORT_R2.md → position_report.md
4. completed/round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md → gemmascope_reconstruction_*_summary.md
5. original_paper_evals/README.md → original_paper_evals/plans/no_training.md → original_paper_evals/plans/training_related.md
```

### 只关心结论（~20 分钟）

只看这三份：

1. `completed/critique/CRITIQUE.md` §1（什么是有趣的，什么是 trivial）
2. `completed/round2_position/SYNTHESIS_REPORT_R2.md` §1, §2, §4（R2 的 Executive Summary + Integrated Analysis）
3. `completed/round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md` §1-3（R3 的论文阅读要点 + 总体量化结果）

---

## 完整文件清单

### 主报告目录：`docs/exp/`

| 文件 | 大小 | 最后修改 | 层级 |
|---|---|---|---|
| `architecture/技术架构与实验流程详解.md` | 36 KB | May 21 | 0 — 前置知识 |
| `completed/round1_baseline/实验报告_R1.md` | 44 KB | May 21 | 1 — Round 1 |
| `completed/round1_baseline/SYNTHESIS_REPORT.md` | 38 KB | May 20 | 1 — Round 1 英文综合 |
| `completed/round1_baseline/quant_report.md` | 3.4 KB | May 21 | 1 — Round 1 定量数据 |
| `completed/critique/CRITIQUE.md` | 25 KB | May 20 | 2 — 批判与转向 |
| `completed/critique/NEXT_STEPS.md` | 10 KB | May 21 | 2 — 下一步决策 |
| `completed/round2_position/实验报告_R2.md` | 47 KB | May 21 | 3 — Round 2 |
| `completed/round2_position/SYNTHESIS_REPORT_R2.md` | 48 KB | May 21 | 3 — Round 2 英文综合 |
| `completed/round2_position/position_report.md` | 1.3 KB | May 21 | 3 — 位置分辨定量数据 |
| `original_paper_evals/README.md` | — | Jun 11 | 6 — 后续正式评估命名空间 |
| `original_paper_evals/plans/no_training.md` | — | Jun 11 | 6 — 无训练评估计划 |
| `original_paper_evals/plans/training_related.md` | — | Jun 11 | 6 — 训练相关评估计划 |

### 团队报告目录：`docs/exp/completed/team_reports/`

| 文件 | 大小 | 最后修改 | 层级 |
|---|---|---|---|
| `NLA_Qwen_批判式复现实验报告.md` | 7.4 KB | May 22 | 4 — 团队汇报 |
| `00_experiment_director_critique.md` | 4.5 KB | May 22 | 4 — 批判 |
| `01_reproduction_quant_baseline.md` | 3.6 KB | May 22 | 4 — 定量基线 |
| `02_limitations_failure_modes.md` | 4.6 KB | May 22 | 4 — 局限性 |
| `03_hidden_state_detection.md` | 4.7 KB | May 22 | 4 — 隐藏状态检测 |
| `04_safety_dynamics_position_resolved.md` | 4.2 KB | May 22 | 4 — 安全动态 |
| `README.md` | 1.1 KB | May 22 | 4 — 目录说明 |

### Original Cases L20 目录：`docs/exp/completed/round3_original_cases/`

| 文件 | 大小 | 最后修改 | 层级 |
|---|---|---|---|
| `ORIGINAL_CASES_L20_汇总报告.md` | 45 KB | May 23 | 5 — Round 3 汇总 |
| `scores_by_condition.md` | 12 KB | May 22 | 5 — 条件评分矩阵 |
| `gemmascope_reconstruction_metrics_all_summary.md` | 5.2 KB | May 22 | 5 — GemmaScope 全量汇总 |
| `gemmascope_reconstruction_metrics_ABC_summary.md` | 4.0 KB | May 22 | 5 — A/B/C 组汇总 |
| `gemmascope_reconstruction_metrics_DEF_summary.md` | 3.8 KB | May 22 | 5 — D/E/F 组汇总 |

### 项目文档目录：`docs/`

| 文件 | 大小 | 最后修改 | 层级 |
|---|---|---|---|
| `qwen_nla_reproduction_detailed_2026-05-20.md` | 11 KB | May 20 | 0 — 环境搭建 |
| `exp/README.md` | — | Jun 04 | 0 — 实验总览 |
| `exp/completed/configs/original_cases_l20_RUNBOOK.md` | 5.3 KB | May 22 | 5 — 运行手册 |
| `exp/completed/configs/original_cases_l20.yaml` | — | May 22 | 5 — 实验配置 |
| `exp/completed/round3_original_cases/next_round_layer20_original_cases_limitations.md` | 15 KB | May 22 | 5 — 下一轮计划 |
| `exp/completed/round3_original_cases/gemmascope_reconstruction_original_cases_l20_report.md` | 7.8 KB | May 22 | 5 — GemmaScope 方法论 |

---

## 核心数据文件位置

如需自己分析原始数据：

| 数据 | 路径 | 行数 |
|---|---|---|
| R1+R2 定量评分 | `completed/round1_baseline/quant_scores.csv` | 391 |
| R3 全量评分 | `completed/round3_original_cases/scores.csv` | 3,315 |
| R3 逐条件汇总 | `completed/round3_original_cases/scores_by_condition.md` | — |
| R3 GemmaScope 全量指标 | `completed/round3_original_cases/gemmascope_reconstruction_metrics_all.csv` | 3,315 |
| R3 Claim Review | `completed/round3_original_cases/claim_review.csv` | 10,939 |
| R3 重建向量 (A/B/C) | `completed/round3_original_cases/gemmascope_reconstruction_metrics_ABC_recon_vectors.npy` | — |
| R3 重建向量 (D/E/F) | `completed/round3_original_cases/gemmascope_reconstruction_metrics_DEF_recon_vectors.npy` | — |
| 实验配置文件 | `completed/configs/` | — |
| 运行手册 | `completed/configs/original_cases_l20_RUNBOOK.md` | — |

---

## 关键指标速查

| 指标 | 全称 | 含义 | 方向 |
|---|---|---|---|
| **cos** | Cosine Similarity | NLA 解码文本嵌入与参考嵌入的余弦相似度，衡量重建语义质量 | 越高越好（上限 ~0.95） |
| **MSE** | Mean Squared Error（缩放） | AR Critic 的方向 MSE，缩放因子 √d_model = 59.87 | 越低越好 |
| **norm** | L2 Norm | 残差流激活向量在采样位置的 L2 范数 | 低值可能表示表征塌缩 |
| **L0 proxy** | — | AR tokenizer 下的 explanation token 数，作为解释成本的代理指标 | — |
| **FVU** | Fraction of Variance Unexplained | Scale-matched AR 重建的 SSE / mean-baseline SSE | 越低越好 |
| **delta LM loss** | — | 替换激活向量后的 next-token CE 增量，最接近"模型行为是否保持" | 越低越好，p50 通常 < 0.001 |

---

## 参考来源

- Anthropic Blog: https://www.anthropic.com/research/natural-language-autoencoders
- Paper: https://transformer-circuits.pub/2026/nla/index.html
- Local PDF: `Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.pdf`
- Gemma Scope: https://arxiv.org/html/2408.05147v2
- Qwen-Scope: https://arxiv.org/html/2605.11887v1
