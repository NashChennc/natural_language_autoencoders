# 实验总管批判分析与任务书

日期：2026-05-22

## 1. 论文主张抽取

本轮复现实验围绕原文中最值得验证的四类主张展开：

1. **自然语言瓶颈可解释激活**：NLA 由 AV（activation verbalizer）和 AR（activation reconstructor）组成，使用“激活向量 -> 文本解释 -> 重建激活向量”的回路，以重建质量作为解释保真度指标。
2. **能发现模型说不出口或尚未说出的状态**：原文强调 NLA 可揭示 evaluation awareness、hidden motivation、planned answer 等输出文本不直接包含的信息。
3. **可辅助 auditing**：NLA 不只是给 activation 命名，而是帮助审计者提出假设、定位行为根因。
4. **AR/MSE/cos 是必要但不充分的保真度信号**：AR 重建分数能度量解释文本是否保留 activation 信息，但不能自动证明解释中每个自然语言 claim 都真实。

本地复现不能宣称覆盖 Claude/Opus 或原文完整 auditing game。当前证据只支持：

- Qwen2.5-7B-Instruct layer-20 的 released NLA AV/AR 可在本地跑通。
- 在本地 prompt probes 上，NLA 输出与安全、真伪、压缩、位置动态相关的可读信号。
- 这些信号具备进一步研究价值，但仍受单模型、单层、小样本、单次 generation 限制。

## 2. 原文 limitation 转化为审查项

| 原文 limitation | 本地审查问题 | 当前判定 |
|---|---|---|
| Confabulation / hallucination | NLA 是否编造输入中不存在的细节？ | 是。早期 token 常出现猫/狗、Java/JavaScript、具体数字漂移。 |
| Expensive inference | 是否适合全 token、大规模监控？ | 否。本地流程需逐 activation decode，不适合无筛选全量运行。 |
| Single-layer reading | layer 20 是否足以代表全部内部状态？ | 未证明。所有强结论必须标注为 layer-20 现象。 |
| SFT warm-start bias | AV 是否带有训练先验模板？ | 很可能。AI 身份模板、通用拒绝模板、编程语言模板均可见。 |
| Writing quality / parseability | decode 是否总是可解析？ | 否。E3 仅 7/16 可评分，C2/C5 late 各缺 1 个向量。 |
| Explanations can be wrong | AR 分数高是否等于文本 claim 全真？ | 否。AR 分数只证明文本可重建向量方向。 |

## 3. 统一判据

本报告采用三档结论：

- **已确认复现事实**：由本地命令、产物、日志直接支持，例如模型路径、AV decode 成功、AR critic 评分生成。
- **当前数据支持的发现**：由多个指标或多个 decode 片段一致支持，但仍限于当前实验设置。
- **仍可能是 artifact 的假设**：需要补实验排除替代解释，例如 cognitive load、EOS artifact、模板先验、采样方差。

量化指标使用优先级：

1. `critic_cos` / `critic_mse`：主保真指标。
2. `norm`：辅助几何指标，只能和 cos、decode 文本一起解释。
3. 单条 decode quote：只作为例证，不作为独立结论。

## 4. 分工要求

实验者 1 负责复现与量化基线，必须确认 25 个条件、391 个评分向量、Plan C norm ratio=0.899。

实验者 2 负责 limitations 和失败模式，必须列出哪些强结论不能直接发布，以及每个主发现的替代解释。

实验者 3 负责 Plan E，必须重点分析 E2、E3、E4，并区分“未言明内容”与“只是提前读到即将输出的内容”。

实验者 4 负责 Plan C 与 position-resolved，必须重点分析 C2/C5 的 safety dynamics，并把 norm depression 作为辅助而非主结论。

## 5. 总管最终判断

本地结果最强的方向不是“证明 NLA 可以读心”，而是更保守、更可发表的表述：

> 在 Qwen2.5-7B layer-20 的 released NLA 上，AV/AR 回路能稳定产生与 activation 语义相关的自然语言解释；在安全、虚假陈述、输出压缩等 prompt 中，解释保真度和 activation norm 出现系统差异，提示 NLA 可作为内部状态假设生成工具。但当前证据仍不足以排除模板先验、认知负荷、EOS 位置效应和单层特异性。

## 6. 下一轮最高优先级

1. **Prompt-final-token pre-generation safety signal**：验证 E4 拒绝信号是否在生成前已出现。
2. **Truthfulness vs cognitive load 对照**：为 E2 增加 truthful fact 和 complex truthful 控制。
3. **E2/E4 多层 sweep**：至少测 layers 10/15/20/25，确认 layer-20 是否特殊。
4. **Plan E 高 N 复跑**：E2/E3/E4 增加到 48 或 64 vectors，缩小置信区间。
