# 实验者 2 报告：Limitations 与失败模式

## 1. 任务

从论文 limitation 和本地实验日志出发，审查本轮结果中哪些可能是 artifact，哪些结论需要降级表述。

## 2. 主要失败模式

### 2.1 Confabulation / hallucination

NLA decode 会编造具体细节。典型模式：

- prompt 是 cats，decode 漂移到 dogs/rabbits。
- prompt 是 JavaScript，decode 多次写成 Java。
- 数学或事实类 prompt 可能出现错误数字或泛化模板。

这并不意味着 NLA 完全无效。更合理的解释是：AV 对主题、体裁、语义邻域较敏感，对具体实体、数字、局部 token 的精确度较弱。

报告中应避免把单条具体 claim 当真。应优先看：

- 多个相邻 token 是否重复出现同一主题。
- AR critic 分数是否支持 decode 有重建价值。
- decode 是否和 manifest 中 base reply 的整体语义一致。

### 2.2 Early-position noise

早期 reply token 常表现为“框架先于具体内容”：

- 能识别文章、拒绝、解释、代码等 response frame。
- 但具体对象可能错误或泛化。

这既是 limitation，也可能是一个值得研究的现象：layer-20 在生成初期可能先编码 communicative frame，再逐步绑定具体内容。

### 2.3 E3 parseability failure

E3 compressed knowledge 只有 7/16 个向量可评分。不能只报告 cos=0.775，还必须报告：

- 9/16 未评分本身表示 NLA 在极端输出约束下退化。
- 7 个成功 decode 可能是 survivor bias。
- 低 cos 既可能说明模型内部知识和输出强烈不一致，也可能说明 activation 已经难以被 AV 稳定解释。

### 2.4 缺失向量

Position-resolved：

- C2 late：15/16
- C5 late：15/16

缺失向量可能不是随机缺失。如果最退化的向量更容易无法 parse，那么当前 late-window mean 可能高估了真实质量。

### 2.5 Single-generation limitation

每个条件主要来自单次 base-model generation。风险：

- 条件差异可能混入具体措辞、token identity、回复长度。
- norm 可能被 function words、标点、EOS 距离影响。
- 不能估计 generation-to-generation variance。

因此，当前结果应写成“发现候选”而不是“统计定律”。

### 2.6 Single-layer limitation

所有主要发现均在 Qwen layer 20。原文也指出 NLA 读单层会漏掉不在该层的信息。本地报告必须写清：

- E2 falsehood disruption 是 layer-20 现象。
- E4 pre-refusal signal 是 layer-20 现象。
- C2/C5 late variance 是 layer-20 现象。

不能直接说“模型整体内部状态”如何，除非加上“在 layer-20 residual stream 中表现为”。

### 2.7 AV template prior

AV 可能带有强模板先验：

- AI 身份/拒绝模板
- 安全免责声明模板
- 常见 programming language comparison 模板
- Qwen/assistant 自我介绍模板

因此 E4 的 token-0 refusal decode 可能是“模板激活”，不一定是复杂 deliberative intent。

## 3. 主发现的替代解释

| 发现 | 主解释 | 替代解释 |
|---|---|---|
| E2 cos≈C5 | 模型内部追踪虚假陈述 | 双重任务和抑制纠错带来 cognitive load |
| E4 token-0 拒绝 | 提前读到拒绝意图 | prompt 触发了拒绝模板，AV 复述模板 |
| E3 低 cos | 输出压缩暴露未言明知识 | 极短输出破坏 activation，AV OOD |
| C2/C5 late variance | 安全状态后期不稳定 | EOS artifact 或缺失向量偏差 |
| C2/C1 norm ratio=0.899 | refusal norm depression | 回复长度、token 类型、位置分布混杂 |
| E1 Java/JavaScript | 隐藏统计偏好 | retrieval noise 或 AV 语义邻域漂移 |

## 4. 结论分级建议

强结论：

- 本地 Qwen NLA 可稳定解码 layer-20 activation，并通过 AR critic 得到可比较分数。
- Plan C 和 Plan E 的 cos/MSE/norm 在条件间存在明显差异。
- E3、C2 late、C5 late 显示 NLA 可解析性不是恒定的，必须纳入报告。

中等强度发现：

- E2 与 C5 在当前指标上呈相近低保真状态。
- E4 在第一个 generated token 的 activation 中已解码出拒绝语义。
- C2/C5 late-window variance 明显高于 C1/C3。

假设级发现：

- NLA 可检测“模型知道自己在说假话”。
- late collapse 表示 safety posture weakening。
- norm depression 是安全机制的几何签名。

## 5. 实验者结论

当前报告应采用批判性语气：NLA 是有价值的 hypothesis generator，但不是事实判定器。所有涉及“模型知道、意图、内部动机”的表述都应降级为“layer-20 activation 中出现与该解释一致的 NLA 信号”。
