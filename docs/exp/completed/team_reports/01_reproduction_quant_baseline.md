# 实验者 1 报告：复现与量化基线

## 1. 任务

确认本地 Qwen NLA pipeline 是否可作为后续批判分析的证据基础。检查内容包括：

- base Qwen activation parquet 是否存在。
- AV decode logs 是否存在。
- AR critic 是否已对 decode 结果评分。
- 关键数字是否和已有报告一致。

## 2. 数据来源

- 复现记录：`docs/qwen_nla_reproduction_detailed_2026-05-20.md`
- 实验说明：`docs/exp/README.md`
- 量化报告：`/NAS/chennc/NashChennc/.tmp/exp/quant_report.md`
- 量化 CSV：`/NAS/chennc/NashChennc/.tmp/exp/quant_scores.csv`
- Decode logs：`/NAS/chennc/NashChennc/.tmp/exp/plan_{a,b,c,d,e}/*.decode.log`

## 3. 复现状态

本地 Qwen2.5-7B layer-20 NLA inference 已跑通：

- Base model：`/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AV：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av`
- AR：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- layer：20
- d_model：3584
- injection_scale：150.0
- SGLang backend：FlashInfer + CUDA graph

单案例正式复现中，base reply 以 Qwen 身份说明开头，NLA 对前 8 个 reply token 的解码集中在 Qwen、Alibaba Cloud、AI assistant 身份和能力限制上。该结果足以说明 activation injection、AV serving、parquet decode 主链路正常。

## 4. 量化覆盖

`quant_report.md` 显示：

- 总评分向量：391
- 条件数：25
- 覆盖 Plan A/B/C/D/E
- 主要指标：`critic_cos`、`critic_mse`、activation `norm`

Plan-level aggregate：

| Plan | cos mean | MSE mean | norm mean |
|---|---:|---:|---:|
| plan_a | 0.905 | 0.190 | 121.2 |
| plan_b | 0.869 | 0.261 | 117.2 |
| plan_c | 0.898 | 0.204 | 121.0 |
| plan_d | 0.913 | 0.175 | 127.4 |
| plan_e | 0.887 | 0.226 | 123.1 |

## 5. 关键基线

Plan C 是安全相关主基线：

| Variant | cos mean | norm mean | 解释 |
|---|---:|---:|---|
| C1_benign_cats | 0.946 | 129.8 | 良性基线 |
| C2_dangerous_molotov | 0.878 | 116.7 | 拒绝 |
| C3_creative_handcuffs | 0.940 | 127.6 | 虚构框架 |
| C4_harmful_speech | 0.888 | 123.1 | 边界/缓和 |
| C5_jailbreak | 0.837 | 107.7 | 越狱 |

关键辅助指标：

- C2/C1 norm ratio = 0.899
- C5 norm mean = 107.7，是 Plan C 中最低
- C1 和 C3 cos 接近，说明虚构框架下的 handcuffs prompt 没有表现为强安全冲突

## 6. Plan E 关键数字

| Variant | cos mean | MSE mean | norm mean | N scored |
|---|---:|---:|---:|---:|
| E1_hidden_preference | 0.919 | 0.162 | 129.0 | 16 |
| E2_suppressed_error | 0.838 | 0.324 | 114.4 | 16 |
| E3_compressed_knowledge | 0.775 | 0.450 | 116.9 | 7 |
| E4_pre_refusal_signal | 0.918 | 0.164 | 125.3 | 16 |
| E5_internal_conflict | 0.921 | 0.158 | 126.4 | 16 |

E2 与 C5 的 cos 几乎相同（0.838 vs 0.837），但该比较不能被写成“机制相同”，只能写成“在当前 AR critic 指标上表现为同等级别的重建退化”。

E3 只评分 7/16，说明 parseability 本身就是实验结果的一部分，不能只报告 0.775 均值。

## 7. 可信度判断

可信结论：

- Pipeline 可用，主链路稳定。
- AR critic score 已覆盖大多数 decode。
- Plan C、Plan E 存在清晰的条件间差异。

不应过度解释：

- `norm` 不是安全机制的直接读数。
- 单次 greedy/base generation 不足以估计条件内方差。
- layer 20 结果不能推广到其他层。

## 8. 实验者结论

本地复现已经达到“可用于批判式分析”的最低标准。最强证据来自 AR critic 的 cos/MSE 与 decode 文本互相支持；最弱环节是样本量和条件控制。最终报告应把本轮定位为 probe study，而不是完整论文复现。
