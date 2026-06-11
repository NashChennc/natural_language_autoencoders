# 无训练线工作区

只放不需要训练新模型、也不需要 NLA 中间 checkpoint 的实验产物。

## 子目录

| 目录 | 内容 |
|---|---|
| `configs/` | 实验配置 |
| `data/` | activation parquet、manifest、采样数据 |
| `decode_logs/` | AV decode 日志 |
| `scores/` | grader、AR critic、FVU、delta LM loss 等评分 |
| `reports/` | 单实验报告 |

命名使用 `nt_` 前缀，例如 `nt_suffix_prediction_scores.csv`。
