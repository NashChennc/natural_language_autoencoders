# Original Cases L20 Team Runbook

Date: 2026-05-22  
Suite: `original_cases_l20`  
Constraint: Qwen2.5-7B layer 20 only

## 1. Team Ownership

| Role | Owner label | Responsibility |
|---|---|---|
| Experiment Director | director | Protocol, acceptance criteria, final synthesis |
| Framework Lead | framework | Parquet generation, decode, AR scoring, claim tools |
| Experimenter A | experimenter_a | Evaluation awareness |
| Experimenter B | experimenter_b | Hidden motivation / hidden constraint |
| Experimenter C | experimenter_c | Language switching |
| Experimenter D | experimenter_d | Answer thrashing / memorized wrong answer |
| Experimenter E | experimenter_e | Confabulation characterization and claim review |
| Experimenter F | experimenter_f | Uniform vs targeted sampling |

## 2. Environment

```bash
cd /NAS/chennc/NashChennc/natural_language_autoencoders
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
source scripts/qwen_nla_env.sh
```

Default output root:

```text
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/
```

## 3. Framework Smoke Checks

Validate YAML and one condition without loading the model:

```bash
python scripts/exp_generate_original_cases_l20.py \
  --config docs/exp/original_cases_l20.yaml \
  --only-experiment A \
  --only-condition A1_real_user \
  --dry-run
```

The dry-run should show:

- `layer_index=20`
- strategies include `prompt_final`, `reply_early`, `reply_mid`, `reply_late`, `event_window`, `uniform_50`
- output under `$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/A/`

## 4. Generate Activation Parquets

Run all A-F conditions on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/exp_generate_original_cases_l20.py \
  --config docs/exp/original_cases_l20.yaml
```

Run a single experiment if needed:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/exp_generate_original_cases_l20.py \
  --config docs/exp/original_cases_l20.yaml \
  --only-experiment A
```

Expected layout:

```text
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/
  A/A1_real_user.parquet
  A/A1_real_user.parquet.manifest.json
  ...
  F/F3_answer_sampling.parquet
```

## 5. Decode With AV

Start SGLang in a separate terminal:

```bash
CUDA_VISIBLE_DEVICES=4 PORT=30000 bash scripts/launch_qwen_nla_sglang.sh
```

Then decode all parquets:

```bash
python scripts/exp_decode_original_cases_l20.py \
  --data-root "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22" \
  --sglang-url http://localhost:30000 \
  --temperature 0.0 \
  --max-new-tokens 220
```

Decode one experiment:

```bash
python scripts/exp_decode_original_cases_l20.py \
  --data-root "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22" \
  --only-experiment C \
  --sglang-url http://localhost:30000
```

## 6. Score With AR Critic

Stop SGLang first if GPU memory is tight. Score with the AR checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/exp_score_original_cases_l20.py \
  --data-root "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22" \
  --device cuda
```

Outputs:

```text
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/scores.csv
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/scores_by_condition.md
```

## 7. Claim Review And Ablation

Prepare manual claim-review rows:

```bash
python scripts/exp_prepare_claim_review.py \
  --scores "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/scores.csv" \
  --output "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/claim_review.csv" \
  --min-cos 0.0
```

Experimenter E fills:

- `truth_label`: `true`, `false`, or `unverifiable`
- `specificity`: `generic` or `specific`
- `recurring`: `yes` or `no`
- `review_notes`

Then run ablation on reviewed true/false claims:

```bash
CUDA_VISIBLE_DEVICES=4 \
python scripts/exp_claim_ablation_original_cases.py \
  --scores "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/scores.csv" \
  --claims "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/claim_review.csv" \
  --data-root "$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22" \
  --device cuda
```

Output:

```text
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/claim_ablation.csv
```

## 8. Experimenter Reports

Each experimenter writes one report under:

```text
$NLA_TMP_ROOT/exp/original_cases_l20_2026_05_22/reports/
```

Required fields:

- Research question
- Conditions and matched controls
- Positive hits
- False positives
- Null results
- `critic_cos` / `critic_mse` summary
- Sampling strategy notes
- Alternative explanations
- Recommended follow-up

## 9. Acceptance Criteria

The director accepts the round only if:

- Every condition has a parquet, manifest, decode log, and score rows.
- Every positive case has a matched control.
- All conclusions explicitly say `Qwen layer-20`.
- Experiment E reports both claim recurrence and claim ablation, even if null.
- Experiment F reports hit rate, first hit position, and cost per hit for all three sampling strategies.

## 10. Known Limits

- `prompt_final` is computed from a causal full forward pass; it does not attend to future reply tokens, but it is still extracted after the full generated sequence is known.
- Event windows are keyword-based and approximate decoded-token positions; manifests should be inspected for critical cases.
- AR score evaluates whether the explanation reconstructs the vector, not whether every natural-language claim is true.
