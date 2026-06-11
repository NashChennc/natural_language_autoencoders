# 原文后半段正式评估命名空间

这个目录专门承接 Anthropic NLA 原文后半段正式评估的本地复现与后续实验。它和 `../completed/` 分开：历史 round 是已经完成的 probes 和报告；这里是后续标准化评估工作区。

## 目录约定

```text
original_paper_evals/
├── README.md
├── plans/                 # 实验设计文档
│   ├── no_training.md
│   └── training_related.md
├── shared/                # 跨实验共享规范
│   └── schema.md
├── templates/             # 新实验报告与 run manifest 模板
│   ├── experiment_report.md
│   └── run_manifest.md
├── no_training/           # 只依赖现有 released AV/AR 的实验线
│   ├── configs/
│   ├── data/
│   ├── decode_logs/
│   ├── scores/
│   └── reports/
└── training_related/      # 依赖 checkpoint/SAE/oracle/target organism 的实验线
    ├── checkpoints/
    ├── configs/
    ├── data/
    ├── decode_logs/
    ├── scores/
    └── reports/
```

## 命名规则

实验 ID 使用稳定 snake_case，前缀标明路线：

| 路线 | 前缀 | 示例 |
|---|---|---|
| 无训练线 | `nt_` | `nt_suffix_prediction` |
| 训练相关线 | `tr_` | `tr_training_curves` |
| 共享/模板 | `shared_` | `shared_schema` |

每个实验的文件名推荐：

```text
configs/{experiment_id}.yaml
data/{experiment_id}.parquet
decode_logs/{experiment_id}.decode.log
scores/{experiment_id}_scores.csv
reports/{experiment_id}.md
```

多 run 时添加日期或 run id：

```text
scores/{experiment_id}_{YYYYMMDD}_{run_id}_scores.csv
reports/{experiment_id}_{YYYYMMDD}_{run_id}.md
```

## 当前入口

- [无训练线计划](plans/no_training.md)
- [训练相关线计划](plans/training_related.md)
- [共享 schema](shared/schema.md)
- [实验报告模板](templates/experiment_report.md)
- [运行 manifest 模板](templates/run_manifest.md)
