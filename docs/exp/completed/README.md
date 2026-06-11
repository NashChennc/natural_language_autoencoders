# 已完成实验归档

这里放 2026-05-20 到 2026-05-23 已完成的 Qwen layer-20 NLA 实验、批判报告、团队报告、配置和旧 LaTeX 综合报告。后续原文正式评估请放到 `../original_paper_evals/`。

## 目录

| 目录 | 内容 |
|---|---|
| `round1_baseline/` | Round 1 定性/定量基线，Plan A-E，391 个评分向量 |
| `critique/` | Round 1 后的方法论批判和 R1 -> R2 决策 |
| `round2_position/` | Round 2 隐藏状态检测和 position-resolved 安全动态 |
| `round3_original_cases/` | Round 3 原文 case 近似复现，A-F 组，3315 行评分 |
| `team_reports/` | R1/R2 面向汇报的主题拆分报告 |
| `configs/` | 已完成实验使用的 YAML 配置和 runbook |
| `latex_report/` | 旧 LaTeX 综合报告及构建产物 |
| `_obsolete/` | 被完整 run 取代的旧 smoke test / brief |

## 使用建议

- 查结论：先读 `team_reports/NLA_Qwen_批判式复现实验报告.md`。
- 查完整结果：读 `round3_original_cases/ORIGINAL_CASES_L20_汇总报告.md`。
- 查原始评分：看 `round3_original_cases/scores.csv` 和 `round3_original_cases/gemmascope_reconstruction_metrics_all.csv`。
- 查旧配置：看 `configs/original_cases_l20_RUNBOOK.md`。
