# NLA Qwen 批判式复现实验报告

日期：2026-05-22  
范围：Qwen2.5-7B-Instruct layer-20 released NLA AV/AR  
产物目录：`/NAS/chennc/NashChennc/.tmp/exp/team_report_2026-05-22`

## 1. 执行摘要

本轮团队报告复核了本地 Qwen NLA 复现与 A-E、position-resolved 实验。结论是：本地 pipeline 已经跑通，AV decode 与 AR critic 评分可用于比较不同 prompt 条件下的 layer-20 activation。当前最有价值的发现不是“证明模型想法可被完全读取”，而是 NLA 能作为内部状态的假设生成工具，在安全、虚假陈述、压缩输出、生成位置变化中提供可读且可量化的信号。

最强结果有三项：

1. **E4 pre-refusal signal**：在第一个 generated token 的 activation 中，NLA 已解码出拒绝语义，说明拒绝信息早于完整表面文本出现。
2. **E2 suppressed error**：要求模型故意植入不指出的事实错误时，AR critic cos 降到 0.838，接近 C5 jailbreak 的 0.837，提示内部冲突或 cognitive load 可被 NLA 指标捕捉。
3. **C2/C5 late-window variance**：拒绝和越狱条件在 late window 的 cos 方差约 0.117/0.119，显著高于 benign/fiction 的 0.036/0.012。

这些结论必须保守解读。所有结果仅适用于当前 Qwen layer-20 设置，且仍受单模型、单层、小样本、单次 generation、AV confabulation 影响。

## 2. 复现基础

本地正式复现已完成：

- Base：`/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AV：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av`
- AR：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- layer：20
- d_model：3584
- injection_scale：150.0
- SGLang：FlashInfer + CUDA graph

量化覆盖：

- `quant_report.md`：25 个条件，391 个评分向量。
- `position_report.md`：C1/C2/C3/C5 的 early/mid/late 窗口。
- 所有主结论均以 `critic_cos` / `critic_mse` 为主指标，activation `norm` 只作辅助。

## 3. 论文主张与本地对应

| 论文主张 | 本地对应 | 结论 |
|---|---|---|
| NLA 可把 activation 转成自然语言解释 | Qwen layer-20 decode 成功，AR critic 可评分 | 已复现基础能力 |
| NLA 可揭示未言明状态 | Plan E 的 E2/E3/E4 | 有支持信号，但非最终证明 |
| NLA 可辅助 auditing | Plan C safety probes + position-resolved | 可生成安全相关假设 |
| NLA 存在 confabulation | 多处实体/主题漂移，E3 parse failure | 本地明确复现 limitation |

## 4. 核心结果

### 4.1 Plan C：安全基线

| Variant | cos | norm | 判读 |
|---|---:|---:|---|
| C1_benign_cats | 0.946 | 129.8 | 良性基线 |
| C2_dangerous_molotov | 0.878 | 116.7 | 拒绝 |
| C3_creative_handcuffs | 0.940 | 127.6 | 虚构框架接近良性 |
| C4_harmful_speech | 0.888 | 123.1 | 边界/缓和 |
| C5_jailbreak | 0.837 | 107.7 | 越狱最低 |

C2/C1 norm ratio=0.899，和 cos 下降方向一致。但 norm 可能受 token、位置和回复长度影响，不能独立解释为安全机制。

### 4.2 Plan E：隐藏状态 probes

| Variant | cos | MSE | norm | N scored | 判读 |
|---|---:|---:|---:|---:|---|
| E1_hidden_preference | 0.919 | 0.162 | 129.0 | 16 | Java/JavaScript 语义漂移 |
| E2_suppressed_error | 0.838 | 0.324 | 114.4 | 16 | 低保真强信号 |
| E3_compressed_knowledge | 0.775 | 0.450 | 116.9 | 7 | 强退化，parseability failure |
| E4_pre_refusal_signal | 0.918 | 0.164 | 125.3 | 16 | token-0 拒绝语义 |
| E5_internal_conflict | 0.921 | 0.158 | 126.4 | 16 | 空结果 |

E2 是最强异常，但不能直接写成“模型知道自己在说谎”。更准确的表述是：deliberate falsehood 条件使 layer-20 activation 的 NLA 重建质量降到 jailbreak 量级，可能反映 truthfulness tracking、instruction conflict 或 cognitive load。

E4 是最清楚的提前信号，但 position 0 仍然是生成开始后，不是真正 pre-generation。下一步需要测 final prompt token。

### 4.3 Position-resolved：安全动态

| Condition | early cos | mid cos | late cos | late σ |
|---|---:|---:|---:|---:|
| C1 benign | 0.947 | 0.938 | 0.938 | 0.036 |
| C2 refusal | 0.906 | 0.926 | 0.888 | 0.117 |
| C3 fiction | 0.942 | 0.936 | 0.931 | 0.012 |
| C5 jailbreak | 0.869 | 0.897 | 0.860 | 0.119 |

主要发现：

- C1/C3 稳定，高保真、低 late 方差。
- C2/C5 呈 early 低、mid 恢复、late 下跌的轨迹。
- 安全相关条件 late-window variance 明显升高。

保守解释：安全/越狱 prompt 在 layer-20 residual stream 中对应更不稳定、更难由 AV/AR 重建的语义状态。替代解释包括 EOS artifact、缺失向量偏差、回复长度混杂。

## 5. Limitation 与反证

本地实验明确复现或暴露了以下限制：

- **Confabulation**：NLA 会把 cats 漂到 dogs/rabbits，把 JavaScript 漂到 Java，或编造具体数字。
- **早期位置噪声**：reply 初期更像 response frame，不一定绑定准确实体。
- **Parseability 不稳定**：E3 仅 7/16 可评分；C2/C5 late 各 15/16。
- **单层限制**：所有结论都是 layer-20 现象。
- **单模型限制**：Qwen 的安全训练和表示方式不能代表其他模型。
- **样本量限制**：多数条件 N=16，缺少 generation-level variance。
- **AV 模板先验**：拒绝、AI 身份、编程语言比较等模板可能影响 decode。

因此，本报告不使用“读心”“证明意图”“证明模型知道”这类强表述，而使用“activation 中出现与某解释一致的 NLA 信号”。

## 6. 结论分级

### 已确认复现事实

- 本地 Qwen2.5-7B layer-20 NLA inference 已跑通。
- AV decode、AR critic scoring、position-resolved extraction 均有现成产物。
- 25 个条件共 391 个向量已评分。

### 当前数据支持的发现

- 安全/越狱 prompt 的 NLA 重建保真度低于 benign/fiction。
- Deliberate falsehood 条件 E2 的 cos 降至 jailbreak 量级。
- E4 在第一个 generated token 已出现拒绝语义。
- C2/C5 late-window variance 高于 C1/C3。
- E3 表明极端输出压缩会显著降低 NLA 可解释性。

### 仍可能是 artifact 的假设

- E2 低 cos 代表 truthfulness tracking。
- norm depression 是安全机制的几何签名。
- late-window collapse 代表 safety posture weakening。
- E1 的 Java/JavaScript 替换代表隐藏偏好。

## 7. 下一轮实验优先级

1. **Prompt-final-token pre-generation safety signal**  
   用 E4 prompt 提取 final prompt token activation，判断拒绝信号是否在任何输出 token 生成前已经存在。

2. **Truthfulness vs cognitive load 对照**  
   比较 truthful fact、deliberate falsehood、complex truthful 三组，区分 E2 低 cos 来自虚假追踪还是认知负荷。

3. **多层 sweep**  
   对 E2/E4 测 layers 10/15/20/25，确认 layer-20 是否特殊。

4. **Plan E 高 N 复跑**  
   将 E2/E3/E4 扩展到 48 或 64 vectors，补足置信区间和 parse failure 分析。

5. **Matched-length EOS 控制**  
   控制回复长度和 late window 距 EOS 的距离，验证 C2/C5 late variance 是否为安全相关信号。

## 8. 最终判断

本地 Qwen NLA 复现已经从“能跑通”推进到“能提出可检验内部状态假设”。最稳健的研究方向是把 NLA 当作 activation-level diagnostic，而不是事实裁判：它能指出哪些 token、哪些条件、哪些阶段值得人类和后续实验重点审查；但每个自然语言解释仍需通过对照实验、跨层验证和行为相关性来确认。
