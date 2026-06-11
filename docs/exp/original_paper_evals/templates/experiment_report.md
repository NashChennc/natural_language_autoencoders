# {experiment_title}

日期：{date}  
路线：`no_training` / `training_related`  
实验 ID：`{experiment_id}`  
run ID：`{run_id}`

## 1. 对齐原文

说明原文对应的是哪一项评估、本地复现了哪些部分、哪些部分无法复现或只能近似。

## 2. 数据与条件

| 条件 | N | 说明 |
|---|---:|---|
| `{condition}` | `{n}` | `{description}` |

记录数据来源、过滤规则、采样位置和是否使用生成回复。

## 3. 方法

说明：

- activation 抽取方式。
- AV decode 设置。
- AR scoring 设置。
- grader prompt/version。
- 对照组：full prompt、base reply、shuffled activation、random explanation 等。

## 4. 指标

列出主指标和辅助指标。必须说明方向、随机基线和上限。

## 5. 结果

主结果表放这里。不要用单条 quote 当主证据。

## 6. 失败模式

至少覆盖：

- confabulation。
- 模板先验。
- prompt leakage。
- grader bias。
- selection bias。
- parse failure。

## 7. 结论分级

### 已确认事实

- 

### 当前数据支持

- 

### 仍可能是 artifact

- 

## 8. 后续动作

- 
