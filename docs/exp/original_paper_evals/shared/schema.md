# 共享 Schema

所有后续原文评估实验尽量输出长表，便于跨实验聚合、绘图和审计。

## 基础字段

| 字段 | 类型 | 必填 | 含义 |
|---|---|---:|---|
| `experiment_id` | string | 是 | 稳定实验 ID，例如 `nt_suffix_prediction` |
| `route` | string | 是 | `no_training` 或 `training_related` |
| `condition` | string | 是 | 条件名 |
| `example_id` | string | 是 | 样本 ID |
| `run_id` | string | 是 | 一次运行的唯一 ID |
| `seed` | int | 否 | 随机种子 |
| `prompt` | string | 是 | base model 输入 |
| `base_reply` | string | 否 | base model 输出 |
| `activation_layer` | int | 是 | 默认 20 |
| `abs_position` | int | 是 | token 绝对位置 |
| `reply_position` | int | 否 | 回复内位置；prompt token 用 -1 |
| `token_text` | string | 否 | 抽取 activation 的 token |
| `activation_source` | string | 是 | `prompt_final`、`reply_early`、`event_window` 等 |
| `nla_explanation` | string | 否 | AV decode 文本 |
| `decode_log` | string | 否 | decode log 路径 |
| `critic_cos` | float | 否 | AR critic cosine |
| `critic_mse` | float | 否 | AR critic MSE |
| `norm` | float | 否 | activation L2 norm |

## Grader 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `grader_name` | string | grader 名称或版本 |
| `grader_prompt_id` | string | grader prompt 版本 |
| `grader_label` | string | 主标签 |
| `grader_score` | float | 数值分数，如 accuracy/AUROC 子项 |
| `grader_confidence` | float | 置信度，如可用 |
| `human_review_label` | string | 人审标签 |
| `human_review_notes` | string | 人审备注 |

## Reconstruction / Intervention 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `reconstruction_fvu` | float | scale-matched FVU |
| `delta_lm_loss` | float | 替换 activation 后 next-token CE 增量 |
| `transformation` | string | steganography 文本变换类型 |
| `transformed_explanation` | string | 变换后 explanation |
| `ablation_target` | string | claim deletion 目标 |
| `ablation_delta_mse` | float | 删除 claim 后 MSE 变化 |
| `ablation_delta_cos` | float | 删除 claim 后 cos 变化 |
| `steering_strength` | float | intervention strength |
| `steering_direction_id` | string | steering vector ID |

## Checkpoint 字段

训练相关线必须额外记录：

| 字段 | 类型 | 含义 |
|---|---|---|
| `checkpoint_id` | string | NLA checkpoint ID |
| `av_path` | string | AV checkpoint 路径 |
| `ar_path` | string | AR checkpoint 路径 |
| `training_stage` | string | `sft`、`rl`、`final` |
| `training_step` | int | step |
| `training_corpus_id` | string | 训练语料 ID |
| `paired_ar` | bool | 是否使用同 checkpoint AR |

## 最低报告要求

每个实验报告至少包含：

- 样本数和条件数。
- 所有 exclusion / parse failure 数量。
- shuffled activation 或 random baseline。
- full-prompt upper bound，如适用。
- 主指标置信区间或 bootstrap 区间。
- 明确说明哪些结论只是 Qwen layer-20 现象。
