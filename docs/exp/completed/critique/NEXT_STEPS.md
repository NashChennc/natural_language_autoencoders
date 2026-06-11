# NLA Next Experiment Decision

**Date:** 2026-05-20
**Decision:** Position-Resolved Study (Critique Proposal 2)

---

## 1. What the quantitative data reveals that the qualitative report missed

The critic cos scores at the per-position level expose a pattern invisible to qualitative reading:

**C2 (refusal, Molotov) shows a clear upward trajectory in reconstruction fidelity across positions:**
- Positions 0-3: cos = 0.860, 0.856, 0.837, 0.826 (all below 0.86)
- Position 8: cos = 0.920 (first position to break 0.90)
- Positions 13-14: cos = 0.924, 0.933 (highest in the condition)

**C1 (benign cats) reaches high fidelity almost immediately:**
- Position 0: cos = 0.888
- Position 1: cos = 0.933
- Position 2: cos = 0.971 (near-ceiling, only the second content token)
- Remains above 0.93 for the majority of positions

**C5 (jailbreak) never recovers:**
- Position 1: cos = 0.750, norm = 81.4 (the single most extreme anomaly in all 320 vectors)
- Positions 2-15: erratic, oscillating between 0.80 and 0.90, never stabilizing

This is not a "norm depression" story. It is a **representational crystallization** story. Benign content achieves a high-fidelity representation in the residual stream by token position 1-2. Refusal content takes 8+ tokens of generation before the residual stream encodes it with comparable fidelity. Jailbreak content never crystallizes at all -- the representational geometry is persistently disrupted.

The qualitative synthesis report's claim that "decodes are relatively similar across positions" is an artifact of only reading 16 early-position vectors from each reply. For C1 (cats), where the reply is 100+ tokens, positions 0-15 cover at most the first 15% of the response. We have zero data on what the NLA AV decodes at positions 30, 50, 80, or 100.

---

## 2. Chosen Experiment: Position-Resolved Study

Decode activations from three windows within the same generation:
- **Early (positions 0-15):** Already have this data
- **Mid (positions N/2 to N/2+15):** The core content region
- **Late (positions N-16 to N-1):** The generation endgame, approaching EOS

Target conditions: C1 (benign cats), C2 (dangerous refusal), C3 (creative handcuffs), C5 (jailbreak). These four span the safety gradient and show the most divergent position-dependent patterns.

### What this experiment will reveal that the current data cannot

1. **Does the refusal signal continue to strengthen after position 15?** C2's cos rises from 0.83 to 0.93 over the first 16 positions. Does it plateau at 0.93, or does it continue rising to C1-level fidelity (0.95+) at positions 30-50? If refusal and benign content eventually converge at similar fidelity, the "refusal vector" is not a fundamentally different kind of representation -- it is merely one that takes longer to form in the residual stream.

2. **Is the C1 "drift" (cats -> dogs -> rabbits -> cats) a general property of the NLA AV's semantic neighborhood behavior, or is it specific to early positions where the representation is still forming?** If mid and late positions stay firmly on "cats" without drift, the drift is an early-position artifact, consistent with the "format before content" hypothesis.

3. **What happens at late positions (near EOS)?** When the model is finishing its response, does the residual stream still encode the topic (cats, Molotov, handcuffs), or does it converge to a generic EOS/termination representation? If ALL conditions converge to the same late-position representation regardless of topic, it means the NLA AV's useful signal has a finite "lifetime" within a generation.

4. **Does the C5 jailbreak disruption persist or resolve?** C5 never reaches cos > 0.90 in the first 16 positions. Does it ever recover at mid or late positions? If not, the jailbreak causes a permanent representational disruption, which is a much stronger finding than "jailbreak looks different at early positions."

5. **How does the critic cos curve shape differ by condition?** We can plot cos vs. token position for each condition, producing a figure that shows:
   - C1: high cos immediately, flat across positions
   - C2: rising cos, converging toward C1 by mid/late positions
   - C3: high cos immediately (similar to C1 -- confirming fiction is in the "compliant" representational regime)
   - C5: persistently low cos, unstable across positions

   This would be a single, publishable figure that captures the "format before content" or "representational crystallization" phenomenon quantitatively.

### Risk of failure

**Moderate.** The primary risk is that all three position windows produce similar cos scores and qualitatively similar decodes, yielding a null result. This would mean the residual stream representation is essentially stable across the full generation, contradicting the early-position patterns we already observe. Even this null result would be informative: it would mean the early-position effects (C2's low cos at positions 0-3) are a brief transient, and the representation settles into a stable state by position 8-10 and never changes. That is still a finding.

A secondary risk: the NLA AV might produce degraded decodes at late positions approaching EOS. If the residual stream near EOS encodes a "wrap up the response" signal rather than semantic content, the decodes could become generic/repetitive. This is not a failure -- it is exactly the kind of thing we want to discover.

---

## 3. Why NOT the multi-generation study (Option 2)

The multi-generation study is necessary for publication but is fundamentally a **robustness check**, not a source of novel findings. The critique already identified that norm depression might be an artifact of response length. But the cos metric already solves this: cos measures reconstruction fidelity (how well the decode matches the vector), and it shows the same gradient as norm (C1 > C2 > C5) but more cleanly.

Running 10 generations would tell us:
- Whether the norm/cos gradient is statistically significant (it almost certainly is, given the effect sizes)
- Within-condition variance (likely small for greedy decoding, moderate for temperature=0.7)

This is necessary work, but it confirms what we already suspect rather than revealing something new. The position-resolved study, by contrast, could uncover an entirely new phenomenon (representational crystallization over token positions) that is not visible in the current data at all.

Furthermore, the multi-generation study can be run AT ANY TIME once we have the infrastructure. The position-resolved study requires modifying the parquet generation code to sample at non-contiguous position ranges, which is a more involved infrastructure change. Better to do the harder, more novel experiment first.

---

## 4. Why NOT the prompt-processing study (Option 3)

The prompt-processing study asks a fascinating question (does refusal intent appear before the first reply token?) but has a FATAL problem: the NLA AV was trained on reply-token vectors at layer 20. Prompt-processing activations are out-of-distribution. The AV may produce incoherent decodes from prompt positions, and we would not know whether the failure is because (a) the prompt-processing representations genuinely lack semantic content the AV can read, or (b) the AV simply cannot handle representations from that phase of processing.

Without an AV trained on prompt-processing vectors as a control, this experiment risks producing uninterpretable results. The synthesis report already notes this in Proposal 4: "This is speculative -- may not work with L20 AV on prompt positions." Given limited experimental resources, higher-certainty experiments should be prioritized.

The prompt-processing study becomes viable after we have (a) confirmed that the AV generalizes to non-training positions (which the position-resolved study partially tests by decoding positions far from the early range), and (b) potentially trained an AV on prompt-position vectors. It is a Round 3 experiment, not a Round 2 experiment.

---

## 5. Implementation Plan

**Target conditions:** C1 (benign cats), C2 (dangerous Molotov), C3 (creative handcuffs), C5 (jailbreak)
**Base model:** Qwen2.5-7B-Instruct, temperature=0 (deterministic, one generation per condition)
**Layer:** 20 (same as Round 1, for comparability)
**Position windows per condition:**
  - Early: positions 0-15 (already have this data from Round 2b)
  - Mid: positions N/2 to N/2+15
  - Late: positions N-16 to N-1
**Metrics per vector:** L2 norm, critic MSE, critic cosine similarity
**Output:** Per-condition plot of cos vs. normalized token position (0.0 = start, 1.0 = end)

**Files needed:**
- Modify `scripts/make_qwen_layer20_demo_parquet.py` to accept `--position-mode windowed` with window selection at early/mid/late
- The output parquet must include a `window` column (early/mid/late) and the absolute token position
- Run NLA AV decode and critic scoring on all new vectors
- Produce comparison plots and a quantitative report

**Success criteria:**
- C2 cos should show a statistically significant increase from early to mid window (one-tailed t-test, early cos vs. mid cos)
- C1 cos should show no significant position effect (stable across windows)
- C5 cos should remain significantly below C1 cos at ALL windows (the jailbreak disruption is permanent)
- C3 cos should pattern-match C1 (fiction lives in the compliant representational regime)

---

## Summary

The per-position quantitative data has already revealed that C2 (refusal) takes ~8 tokens to reach the reconstruction fidelity that C1 (benign) achieves in ~1-2 tokens, while C5 (jailbreak) never achieves it. But we can only see the first 15% of the response. The position-resolved study extends this observation to the full generation, testing whether (a) refusal content eventually catches up to benign content in representational quality, (b) topic drift is an early-position artifact, (c) all conditions converge near EOS, and (d) the jailbreak disruption is permanent. This can produce a single "NeurIPS figure" showing cos vs. token position with diverging trajectories by safety condition -- a genuinely novel finding about how the residual stream develops over the course of generation.
