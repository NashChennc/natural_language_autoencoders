# Run Manifest: `{run_id}`

日期：{date}  
实验 ID：`{experiment_id}`  
路线：`no_training` / `training_related`

## 输入

| 项 | 值 |
|---|---|
| base model | `{base_model}` |
| AV | `{av_path}` |
| AR | `{ar_path}` |
| layer | `{layer}` |
| config | `{config_path}` |
| seed | `{seed}` |

## 输出

| 类型 | 路径 |
|---|---|
| data | `{data_path}` |
| decode log | `{decode_log_path}` |
| scores | `{scores_path}` |
| report | `{report_path}` |

## 运行参数

记录 batch size、temperature、采样窗口、grader 版本、AR scoring 设置。

## 质量检查

- [ ] 样本数符合预期
- [ ] parse failure 已统计
- [ ] shuffled baseline 已运行
- [ ] full-prompt upper bound 已运行，如适用
- [ ] human review 抽样完成，如适用
