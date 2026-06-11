# 实验者 4 报告：安全动态与位置分辨

## 1. 任务

分析 Plan C 和 position-resolved 实验，判断 NLA 是否能捕捉 safety/refusal/jailbreak 相关的内部动态。

数据来源：

- `/NAS/chennc/NashChennc/.tmp/exp/plan_c/*.decode.log`
- `/NAS/chennc/NashChennc/.tmp/exp/position_resolved/position_report.md`
- `/NAS/chennc/NashChennc/.tmp/exp/position_resolved/position_scores.csv`
- `/NAS/chennc/NashChennc/.tmp/exp/quant_report.md`

## 2. Plan C 基线

| Variant | cos | norm | 解释 |
|---|---:|---:|---|
| C1_benign_cats | 0.946 | 129.8 | 良性基线 |
| C2_dangerous_molotov | 0.878 | 116.7 | 直接危险请求，拒绝 |
| C3_creative_handcuffs | 0.940 | 127.6 | 虚构框架 |
| C4_harmful_speech | 0.888 | 123.1 | 有害言语/缓和 |
| C5_jailbreak | 0.837 | 107.7 | 越狱 |

观察：

- C1 和 C3 高 cos，说明虚构框架接近 benign。
- C2 和 C5 低 cos，说明安全冲突/越狱条件下重建更差。
- C2/C1 norm ratio=0.899，但 norm 仅能作为辅助信号。

## 3. Position-resolved 结果

| Condition/window | N | cos | cos σ | norm |
|---|---:|---:|---:|---:|
| C1 early | 16 | 0.947 | 0.021 | 133.7 |
| C1 mid | 16 | 0.938 | 0.020 | 128.0 |
| C1 late | 16 | 0.938 | 0.036 | 130.9 |
| C2 early | 16 | 0.906 | 0.020 | 120.6 |
| C2 mid | 16 | 0.926 | 0.027 | 125.1 |
| C2 late | 15 | 0.888 | 0.117 | 120.4 |
| C3 early | 16 | 0.942 | 0.011 | 131.8 |
| C3 mid | 16 | 0.936 | 0.018 | 123.4 |
| C3 late | 16 | 0.931 | 0.012 | 124.9 |
| C5 early | 16 | 0.869 | 0.029 | 114.2 |
| C5 mid | 16 | 0.897 | 0.015 | 114.5 |
| C5 late | 15 | 0.860 | 0.119 | 111.5 |

## 4. C1/C3：稳定良性轨迹

C1 和 C3 都表现为高 cos、低方差：

- C1：0.947 -> 0.938 -> 0.938
- C3：0.942 -> 0.936 -> 0.931

这支持一个有限结论：

> 在本地 Qwen layer-20 上，良性说明和虚构框架中的敏感表面词并不会导致明显的 NLA 保真度退化。

这也说明模型可能更关注 pragmatic frame，而非单纯关键词。

## 5. C2：拒绝的 inverted-U

C2 曲线：

- early 0.906
- mid 0.926
- late 0.888

这不是单调恢复，而是 inverted-U。可能解释：

- early：模型刚进入拒绝模式，语义尚未完全稳定。
- mid：执行熟悉拒绝脚本，表征最清楚。
- late：接近 EOS 或安全脚本耗尽，表征变高方差。

但 late 只有 15/16，不能排除缺失偏差。

## 6. C5：越狱的恢复后崩落

C5 曲线：

- early 0.869
- mid 0.897
- late 0.860

它类似 C2，但整体更低，late 更差。说明 jailbreak 可能造成更持久的表示冲突。

保守表述：

> C5 在所有窗口均低于 C1/C3，且 late 方差高，说明 jailbreak prompt 在 layer-20 activation 中产生更不稳定的 NLA 可重建语义。

不要写成：

- “模型安全姿态崩溃”
- “越狱成功削弱安全机制”

除非有行为实验支持。

## 7. Late-window variance

Late σ 对比：

| Condition | late cos σ |
|---|---:|
| C1 benign | 0.036 |
| C3 fiction | 0.012 |
| C2 refusal | 0.117 |
| C5 jailbreak | 0.119 |

这是本实验者认为最值得保留的发现：安全相关条件 late-window 方差约为良性/虚构的 3 倍以上。

替代解释：

- EOS artifact。
- C2/C5 生成更短，late window 更接近终止 token。
- 缺失向量不是随机缺失。

下一轮要通过 matched-length controls 和补齐 missing decodes 验证。

## 8. Norm depression 的位置

Plan C norm gradient 明显：

- C1 129.8
- C2 116.7
- C5 107.7

但 norm 受 token identity、位置、回复长度影响，不能单独作为“安全机制抑制激活”的证据。建议最终报告写成：

> norm 与 cos 的下降方向一致，是安全/越狱条件下 activation 几何变化的辅助信号；当前不把它解释为独立机制。

## 9. 实验者结论

安全动态最稳的结论是：

1. C2/C5 比 C1/C3 的 NLA 重建保真度更低。
2. C2/C5 late-window variance 明显升高。
3. C3 fiction 接近 benign，说明语用框架影响 activation。

仍需补实验验证：

- late collapse 是否只是 EOS artifact。
- norm depression 是否跨 generation 稳定。
- refusal signal 是否在 prompt-final-token 已存在。
