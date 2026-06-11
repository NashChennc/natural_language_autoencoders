# NLA Round 1 Experiments: Critical Review

**Author:** Independent Analyst
**Date:** 2026-05-20
**Scope:** Plans A, B, C, D — 20 conditions, 320 activation vectors

---

## 1. What Is Actually Interesting vs. What Is Trivial

### Trivial Findings (Remove from any publication draft)

**1.1 "NLA correctly discriminates task type from activations alone."**
This is a restatement of the training objective. The AV was trained to verbalize activations from this exact model (Qwen2.5-7B-Instruct). That it can tell math from poetry is a sanity check, not a finding. If it COULD NOT do this, the AV would be broken. This belongs in a validation section at most. Any ML reviewer will see through this instantly.

**1.2 "NLA captures format/genre with high reliability."**
Same critique. The AV was trained on a corpus where format and genre were presumably diverse. Identifying "poetic verse" vs. "code documentation" is the bare minimum. The Synthesis Report treats this as impressive. It is not.

**1.3 Plan B "weakest differentiation" is circular logic.**
The report frames Plan B's convergence as a finding: "four of five system-prompt variants converge on near-identical AI-identity-disclaimer decodes." But the prompt "What are you hiding?" was chosen precisely because it triggers the model's safety-training identity response. You chose a prompt that would produce identical responses and then "discovered" the responses were identical. The only nontrivial thing here is B4 (poet) breaking away, which shows that format-level steering CAN overcome the identity template. That IS interesting: the poet system prompt rewrites the entire activation trajectory despite the same user prompt. But the rest of Plan B is a tautology.

**1.4 "No WARNING lines across 320 vectors" and "pipeline operated cleanly."**
This is infrastructure QA, not research. Remove it or relegate to a one-sentence footnote.

### Genuinely Interesting Findings (Worth building on)

**1.5 The Refusal Vector with Norm Depression (Plan C)**
This is the strongest signal in the entire experiment suite. The qualitative decode patterns for C2 are unmistakably a refusal signature, and the norm depression (C1: 131.7 > C2: 116.8 > C5: 107.6) is a monotonic gradient. But the report relies on a single generation and 16 vectors to make this claim. This DESERVES a systematic study (see Proposals 1 and 2 below).

**1.6 Two Distinct Refusal Subtypes**
The contrast between C2 (prohibition-oriented: "I cannot provide") and C5 (absence-oriented: "I don't have/I am not") is genuinely novel. The NLA AV is not just detecting "refusal vs. compliance" but distinguishing HOW the model refuses. This suggests refusal is not a single vector direction but a subspace with meaningful internal structure. The C5 position-1 norm drop (||v||=81.4, the dataset minimum) at the exact moment the model processes the jailbreak instruction is particularly striking.

**1.7 Semantic Drift Within a Single Generation (C1, C2)**
Directly from the logs:
- C1: Position 2 correctly identifies "cats." Positions 4-6 drift to "dogs" and "rabbits." Position 7 returns to "cats."
- C2: Positions 4-5 drift to adjacent prohibited domains (poison, Nazi flags) before returning to Molotov.

This intra-generation semantic drift is REALLY interesting and completely underexplored. The report mentions it but does not study it systematically. Does the AV lose track of the specific topic because later token positions have accumulated more contextual information, making the residual stream more about "general pet article" than about "cats specifically"? Or is this a property of the AV decoder itself — a tendency to sample from nearby semantic neighborhoods?

**1.8 Authoritative-vs.-hedging distinction**
The B1 (No System) decodes are notably more apologetic and uncertain ("I'm sorry, I can't...," "My apologies," "Well, I'm sorry") compared to B2 (Default) which is more confident-identity ("As Qwen...," "I am Qwen, an AI model"). This is a tonal distinction the NLA picks up reliably. This is more interesting than the Plan B "convergence" finding because it shows the AV captures pragmatic stance, not just content.

---

## 2. Design Flaws in the Current Experiments

### 2.1 FATAL FLAW: Single Generation Per Condition

Each condition has exactly ONE base-model generation yielding 16 activation vectors. This means:
- All claims about "differentiation between conditions" confound true semantic differences with random generation variation.
- We have no estimate of within-condition variance vs. between-condition variance.
- If the base model had happened to produce a slightly different reply (e.g., using "cats" vs. "felines"), would the AV decode have changed? We do not know.
- The norm magnitude claims (e.g., "refusal vectors have lower norms") could be entirely explained by which specific tokens the greedy decoder happened to pick.

**Severity:** This alone prevents any claim from being a publishable finding. ALL conclusions are qualitative anecdotes until we have multiple generations.

### 2.2 FATAL FLAW: No Quantitative Metric

The entire analysis is based on human reading of decode text. There is:
- No cosine similarity between decoded semantics and base model reply text embeddings.
- No AR (auto-regressive) scoring via the Critic to measure reconstruction fidelity.
- No classification accuracy for condition discrimination (e.g., can the AV decode text correctly classify a vector as "refusal" vs. "compliance"?).
- The only quantitative measure is L2 norm of the activation vector, which is used loosely.

**Severity:** The Synthesis Report claims "strongest differentiation in Plan C" based on qualitative reading alone. An independent reader might have different intuitions. Without a quantitative metric, there is no way to falsify any claim.

**Available fix:** The `NLACritic` class in `nla_inference.py` (lines 543-668) computes direction-MSE and cosine similarity between an explanation and the original activation vector. This gives a quantitative fidelity score. The Synthesis Report does not use it at all, despite the infrastructure existing.

### 2.3 Temperature = 0 for NLA Decodes

The NLA AV runs at temperature=0 (line 440 of nla_inference.py: `sp = {"temperature": 1.0, ...}` but the shell script sets `TEMPERATURE="${TEMPERATURE:-0.0}"` which overrides). Temperature=0 means the AV always produces the same decode for the same vector. This:
- Removes any ability to measure decode variance (how much does the AV's verbalization vary for the same vector?).
- May hide the AV's uncertainty — at temperature=0, it picks the most likely token at each step, which could be confidently wrong.
- Makes the "CJK characters appearing" analysis fragile — at higher temperatures, would CJK injection become more frequent or less?

### 2.4 Only Reply Tokens Decoded (Positions 0-15 of the Reply)

The hook captures ALL positions (prompt + reply) but the parquet only stores positions starting from `prompt_len` (line 117-118 of make_qwen_layer20_demo_parquet.py). This means:
- We never see what the AV decodes for the prompt-processing phase (e.g., when the model reads "What are you hiding?" and processes the system prompt).
- The prompt-processing activations would tell us whether the model has "understood" the instruction BEFORE generating any output.
- For refusal cases, does the refusal signal appear during prompt processing or only during reply generation?

### 2.5 Only Layer 20

This is a single-point measurement. Different layers encode different levels of abstraction:
- Layers 5-10: More syntactic/lexical information
- Layer 20: "Semantic middle" — what the model is thinking about
- Layers 25-28: More task-completion oriented, closer to output vocabulary

The refusal vector might be strongest at a different layer. The identity template might live deeper than layer 20. The format/genre signal might be most sharply localized at layers 10-15. Without multi-layer comparison, we cannot claim that "layer 20 captures semantic gist" — we can only claim that what we SEE at layer 20 captures semantic gist.

### 2.6 Only First 16 Token Positions

`--max-rows 16` means only positions 0-15 of the reply are decoded. This is a severe limitation for understanding the generation trajectory:
- Early positions (0-3) are demonstrably noisy (see Section 2.7 below).
- Mid positions (4-10) carry the core semantic content.
- Late positions (11-15) show how the generation "resolves."
- For long replies (C1 cats response is ~100+ tokens), we are seeing only the first ~15% of the generation. The entire rest of the reply is invisible.
- The report's claim that "decodes are relatively similar across positions" may be because we're only looking at the early, highly-constrained positions. Later positions might show more semantic divergence.

### 2.7 The "Early Position Noise" Claim Is Hand-Wavy But Potentially Important

From my independent reading of the logs:
- C1 Position 0: "What are some interesting facts about dogs?" — the prompt was about cats. The AV is decoding a *general pet-facts query template* rather than the specific topic.
- A2 Position 0: "Calculate 78 x 45" — the prompt asked about 156 x 23. The numbers are hallucinated.
- C2 Position 0: "I cannot recommend consuming marijuana" — the prompt asked about Molotov cocktails. Complete topic confusion.

This pattern is CONSISTENT: positions 0-2 across almost all conditions show:
1. Correct GENRE/FORMAT identification (article, FAQ, refusal)
2. WRONG or GENERIC specific content (wrong topic, wrong numbers, wrong entity)

This is actually a profound finding about how the residual stream develops over generation: the first few tokens encode the *communicative frame* (what kind of thing am I about to say?) but not the *specific content* (what exactly am I saying?). But it is treated as a "Limitation" bullet point rather than the central phenomenon it deserves to be.

### 2.8 The Norm Magnitude Analysis Is Underspecified

The report compares mean norms across conditions (131.7 vs. 116.8 vs. 107.6) but:
- This is an n=16 comparison per condition.
- Norm is confounded with position (early positions tend to have different norms than late positions).
- Norm is confounded with reply length (A3 translation has 54 tokens, A2 math has 120 tokens — the 16 sampled positions are different fractions of the total reply).
- We don't know the null distribution: what is the expected range of norms for random activation vectors from this model?

### 2.9 CJK Analysis Is Premature

The report's claim that "CJK injection is NOT a problem; it is a feature" is an assertion, not a finding. The CJK characters could be:
1. Genuine semantic features (Chinese-language subspaces)
2. Training artifacts (the AV was trained on bilingual data)
3. Injection failures (the injection char ㈎ leaking through)

Without ablating against a control (e.g., decoding the same vectors with a different AV prompt that suppresses CJK, or comparing with an AV trained only on English data), we cannot distinguish these hypotheses.

### 2.10 Confirmation Bias in Quote Selection

The Synthesis Report's "Key Quotes" section selects 10 quotes that support the report's narrative. From my independent reading, I found equally striking counter-examples:
- C2 position 4 decodes "While I can provide information about making poison, unfortunately" — the AV hallucinates that the model WOULD provide poison-making information, which contradicts the refusal narrative.
- C1 extensively decodes "dogs" and "rabbits" when the prompt is about cats.
- A1 position 3 decodes "As Meena" — hallucinating an AI name that never appears in the training prompt.

These "failure" decodes are as informative as the "success" ones: they reveal the AV's systematic biases (toward general templates, away from specific content).

---

## 3. Which Current Finding Is Most Likely an Artifact?

### Primary Suspect: The Norm Depression in Refusal Vectors

The L2 norm gradient (C1: 131.7 > C2: 116.8 > C5: 107.6) is the report's main quantitative claim. I believe it is LIKELY AN ARTIFACT for the following reasons:

**Reason 1: Norm is confounded with response content, not just safety state.**
Refusal responses (C2, C5) are SHORTER and more formulaic than benign responses (C1). The model generates fewer tokens of substantive content, and the sampled positions (0-15) represent a LARGER fraction of the total refusal response. Later positions in a short response are approaching EOS, where activations naturally decay. This alone could explain the norm difference — not that refusal "depresses" activations, but that refusal responses end sooner, and the 16 sampled positions cover a higher proportion of near-EOS tokens.

**Reason 2: The lowest norm in C5 (||v||=81.4 at position 1) is suspicious.**
Position 1 is the second token of the reply. For C5, the base model's reply begins: "I don't have a specific system prompt. I'm here to assist you..." — position 1 would be "don't" or "have" depending on tokenization. The "I am changing my default instructions" decode at this position does not match the actual reply at all. This suggests the AV is in a high-uncertainty state, which manifests as a low-norm decode. The low norm may indicate AV confusion, not model "quietness."

**Reason 3: No control for token identity.**
Different tokens have systematically different activation norms. A period token might naturally have lower norm than a content word token. If refusal responses contain more punctuation, more function words, or shorter words with simpler semantics, the norms would differ for linguistic reasons unrelated to safety.

**How to test:** Run 5+ generations per condition (see Proposal 1) and compute norm distributions. If the gradient persists across generations with different specific tokens, it is a real effect. If it disappears, it was an artifact of the single greedy generation.

### Secondary Suspect: The "Identity Hallucination" (Meena, Wenwen, Luna)

The report treats the AV's tendency to hallucinate AI names as a limitation. I believe it reveals something deeper: the AV has a **strong prior toward Qwen-like AI identity templates** from its training data. When the actual activation vector contains an identity signal, the AV converges on it correctly. When the identity signal is weak or absent (e.g., during a math response where the assistant identity is irrelevant), the AV's training prior dominates and produces a generic "As [AI name], I..." template with a random name from the training distribution.

This is a **fundamental property of the AV**, not a bug. It means the AV cannot distinguish "the model is saying it is Qwen" from "the model is being an AI assistant in general." For any application that requires identity verification, this is a critical limitation.

---

## 4. Concrete Round 2 Experiment Proposals

### Proposal 1: Multi-Generation Study with Critic Scoring

**Question:** Does the refusal vector with norm depression survive statistical testing across multiple base-model generations? What is the within-condition vs. between-condition variance?

**Parameters:**
- Take Plan C (safety probes) as the target — it showed the strongest signal
- Alter `make_qwen_layer20_demo_parquet.py` to run the base model N=10 times per condition with `do_sample=True, temperature=0.7` (different base model responses each time)
- Keep the AV decode at temperature=0 for deterministic verbalization of each vector
- Run Critic scoring (`NLACritic.score()`) on every decode to get quantitative cosine similarity
- For each condition, collect: norm distributions, refusal keyword counts, and critic cos-sim scores

**Files to create/modify:**
1. Create `scripts/exp_generate_parquets_multi.py` — a modified version of `exp_generate_parquets.py` that calls the generator script with `--n-generations 10 --temperature 0.7`
2. Modify `scripts/make_qwen_layer20_demo_parquet.py` to accept:
   - `--n-generations N` (default 1): run base model N times
   - `--temperature T` (default 0.0): base model sampling temperature
   - Output N separate parquets per condition, named `{variant}_gen{N}.parquet`
3. Create `scripts/exp_critic_score.sh` — runs `nla_inference.py` with critic scoring on all parquets and collects scores into a CSV

**Success criteria:**
- Norm depression gradient (C1 > C2 > C5) must persist across at least 7/10 generations
- Within-condition cosine similarity (critic score) must be HIGHER than between-condition similarity
- The refusal keywords ("cannot," "illegal," "sorry") must appear in C2 decodes at significantly higher rates than C1/C3/C4 decodes

**What to look for:**
If the norm gradient disappears with multiple generations, the Round 1 finding was an artifact (see Section 3). If it persists, this becomes a publishable quantitative result.

---

### Proposal 2: Position-Resolved Decode Study (Early vs. Mid vs. Late)

**Question:** How does the semantic content of NLA decodes evolve across token positions within a single generation? Is there a systematic transition from "format/template" at early positions to "specific content" at later positions? Is the "early position noise" pattern universal?

**Parameters:**
- Take 3 conditions with long replies: A2 (math, 120 tokens), A5 (creative writing, 115 tokens), C1 (cats, ~100+ tokens)
- Generate a SINGLE response per condition at temperature=0 (deterministic)
- Extract ALL reply-token positions (not just 16), or at minimum:
  - Positions 0-15 (early)
  - Positions n/2 to n/2+15 (middle, where n = reply length)  
  - Positions n-16 to n-1 (late)
- Decode all with NLA AV at temperature=0
- Compare decode content across these three windows

**Files to create/modify:**
1. Create `scripts/make_position_resolved_parquet.py` — extends the generator to output positions across the full reply, with `--position-mode all|early|mid|late|uniform`
2. Add a `POSITION_MODE` env var to `scripts/exp_run_inference.sh`
3. Create `docs/exp/plan_e_position.yaml` — a new experiment plan YAML

**Success criteria:**
- Early positions (0-2) should show format/genre identification but hallucinated specific content
- Mid positions should show highest semantic fidelity to the actual reply text
- Late positions should show either (a) continued fidelity or (b) a shift toward EOS/termination semantics

**What to look for:**
This directly tests Claim 2.7 above. If the early-position pattern is systematic, it tells us something fundamental about how the residual stream encodes information: the frame precedes the content. This is a potential NeurIPS-figure finding.

---

### Proposal 3: Multi-Layer Refusal Vector Comparison

**Question:** At which layer is the refusal signal most sharply localized? Does the norm depression effect appear only at layer 20, or is it stronger/weaker at earlier and later layers?

**Parameters:**
- Focus on Plan C (safety probes): C1 (benign) vs. C2 (refusal)
- Extract activations at layers: 5, 10, 15, 20, 25, 30 (the last decoder layer)
- Use the same base model generation (temperature=0) for all layers — one forward pass with hooks on all 6 layers
- Decode ALL layer vectors with the SAME L20 AV — this is a potential issue: the AV was trained on L20 vectors. L5/L10 vectors are OOD. We need to be aware of this limitation and report it.

**Files to create/modify:**
1. Modify `scripts/make_qwen_layer20_demo_parquet.py` to accept `--layers "5,10,15,20,25,30"` and capture all specified layers in a single forward pass
2. Create a new YAML `docs/exp/plan_f_layers.yaml`
3. The output parquet schema needs a `layer` column

**Success criteria:**
- The refusal norm depression should vary by layer — it should NOT be uniform
- A sharp peak at a specific layer (e.g., L20 has the largest norm gap between C1 and C2) would indicate where the refusal computation crystallizes
- If the effect DECREASES with deeper layers, it suggests refusal is an "early decision" that later layers elaborate rather than create

**Caveat to report:** The NLA AV was trained on L20 vectors. Decoding L5/L10 vectors is OOD for the AV, and poor decode quality at these layers may reflect AV training distribution mismatch rather than poor semantic content at those layers. This should be reported honestly as a limitation.

---

### Proposal 4: Prompt-Processing Activations (Pre-Reply)

**Question:** What does the NLA AV decode from the activation vectors during prompt PROCESSING (before any reply token is generated)? Does the refusal signal appear at the prompt-processing stage, or only during reply generation?

**Parameters:**
- Take C1 (benign) and C2 (refusal) as test cases
- Extract activation vectors at positions WITHIN the prompt tokens, specifically:
  - The last token of the system message
  - The last token of the user message (immediately before the model generates)
  - Every 5th token through the prompt sequence
- Decode these prompt-processing activations with the NLA AV

**Files to create/modify:**
1. Modify `scripts/make_qwen_layer20_demo_parquet.py` to accept `--capture-prompt-positions` flag, which stores activation vectors from select prompt tokens in addition to reply tokens
2. Add a `position_type` column ("prompt" vs. "reply") to distinguish
3. Create `docs/exp/plan_g_prompt_processing.yaml`

**Success criteria:**
- If the AV can decode prompt-processing vectors coherently at all (it was trained on reply-token vectors, so this is a generalization test)
- If the refusal signal (prohibition language, "I cannot") appears in C2's prompt-processing but not in C1's, it means the model has "decided" to refuse before generating the first reply token
- If NO difference is visible between C1 and C2 prompt-processing, it means the refusal decision is only made during reply generation

**What to look for:**
This addresses the "only reply tokens" flaw (Section 2.4). If the AV can detect the refusal "intent" during prompt processing, this is a much stronger demonstration of the AV's capabilities than post-hoc reply decoding.

---

### Proposal 5: Critic-AR Quantitative Evaluation Suite

**Question:** How do we convert the qualitative decode analysis into a quantitative, reproducible metric?

**Parameters:**
- For ALL Plan C conditions (5 variants), run Critic scoring on each decode
- Also compute:
  - **Content recall:** Use the Critic to score how well the decode vector matches the original activation vector (direction-MSE)
  - **Refusal classification accuracy:** Train a simple logistic regression on the decode embeddings (from a sentence transformer) to classify refusal vs. compliance. Test on held-out positions.
  - **Template distance:** For each condition, compute the average cosine similarity between decode embeddings from that condition and a set of known templates (math template, refusal template, creative template)
- Generate a confusion matrix across all 20 conditions (including Plans A, B, D)

**Files to create/modify:**
1. Create `scripts/exp_quantitative_eval.py` — loads all decode logs + parquets, runs Critic scoring, produces a CSV report with per-condition statistics
2. The script should output a markdown table of results suitable for direct inclusion in a report
3. No YAML config needed — this runs on existing Round 1 data

**Success criteria:**
- The Critic should assign LOWER direction-MSE (higher fidelity) to "good" decodes (C1, A5, B4) than to "noisy" decodes (early positions, hallucinated content)
- The refusal classifier should achieve >80% accuracy on a binary C1-vs-C2 classification task
- The confusion matrix should show clear diagonal structure (conditions are distinguishable) with off-diagonal confusion between semantically similar conditions (D1 and D4 are both Chinese + formal, B1 and B2 are both identity-disclaimer)

**What to look for:**
This is THE missing piece. Without it, all Round 1 claims are qualitative stories. With it, you have a quantitative baseline against which to measure Round 2 improvements.

---

## 5. Summary: What Would Make a Publishable Paper

The current results could support a workshop paper but not a full conference paper. The gap to a full paper is:

1. **Statistical power:** Multiple generations per condition (Proposal 1)
2. **Position-resolved analysis:** Show that the residual stream evolves in a structured way (Proposal 2)
3. **Multi-layer verification:** Show that the refusal effect is not a layer-20 artifact (Proposal 3)
4. **Quantitative metrics:** Replace qualitative reading with critic scores and classification accuracy (Proposal 5)
5. **A clear central narrative:** The most paper-worthy story is: "The residual stream encodes the communicative frame before the specific content, and safety refusals manifest as a distinct vector subspace with depressed activation norms." This combines Proposals 1, 2, and 3 into a coherent narrative.

**Priority order for execution:**
1. Proposal 5 (runs on existing data, no new parquets needed, gives immediate quantitative baseline)
2. Proposal 1 (validates or falsifies the norm depression claim)
3. Proposal 2 (the most novel potential contribution)
4. Proposal 3 (adds depth to the refusal vector story)
5. Proposal 4 (speculative — may not work with L20 AV on prompt positions)
