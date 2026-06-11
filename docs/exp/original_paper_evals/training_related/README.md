# 训练相关线工作区

只放需要训练产物的实验，包括 NLA checkpoints、SAE、direct activation oracle、trained hidden-objective target organism 等。

## 子目录

| 目录 | 内容 |
|---|---|
| `checkpoints/` | checkpoint manifest、路径清单、训练元数据 |
| `configs/` | 训练相关实验配置 |
| `data/` | activation parquet、训练/评估数据索引 |
| `decode_logs/` | 多 checkpoint decode 日志 |
| `scores/` | training curves、SAE agreement、oracle、auditing scores |
| `reports/` | 单实验报告 |

命名使用 `tr_` 前缀，例如 `tr_training_curves_scores.csv`。
