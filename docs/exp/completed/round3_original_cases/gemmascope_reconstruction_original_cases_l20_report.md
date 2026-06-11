# Gemma Scope-style Reconstruction Metrics for Original Cases L20

Date: 2026-05-22

## Scope

This rerun evaluates the original-case/limitation experiment suite with the Gemma Scope-style quality baseline:

- mean L0
- delta LM loss / delta cross-entropy
- FVU / normalized reconstruction loss

Local outputs:

- Full CSV: `/NAS/chennc/NashChennc/.tmp/exp/original_cases_l20_2026_05_22/gemmascope_reconstruction_metrics_all.csv`
- Full summary: `/NAS/chennc/NashChennc/.tmp/exp/original_cases_l20_2026_05_22/gemmascope_reconstruction_metrics_all_summary.md`
- Script: `scripts/exp_gemmascope_reconstruction_original_cases_l20.py`

Model setup:

- Base LM: `/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AR reconstructor: `/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- Patched layer: `20`
- Rows: `3315`
- Delta LM loss rows: `3256`; final-token rows have no next-token target and are `NA` for delta.

## Metric Comparison

Gemma Scope is the best baseline for SAE quality reporting: it treats mean L0 as sparsity, delta LM loss as the primary behavior-fidelity metric, and FVU as secondary activation-space fidelity. This matches the paper's framing that FVU is reconstruction-only, while delta loss measures the causal effect of splicing reconstructions back into the LM forward pass. See [Gemma Scope](https://arxiv.org/html/2408.05147v2).

For this NLA run, the mapping is not one-to-one:

| Metric | Standard SAE meaning | NLA measurement here | Interpretation |
|---|---|---|---|
| mean L0 | Mean active SAE latents per token | `nla_l0_proxy_tokens`: AR-tokenized explanation length | Not comparable to SAE latent L0; use as explanation-cost proxy only |
| FVU | Reconstruction SSE normalized by always predicting dataset mean | Scale-matched AR reconstruction SSE / mean-baseline SSE | Valid as activation-space fidelity, but NLA AR is direction-trained |
| delta LM loss | CE increase after SAE reconstruction is spliced into LM | Next-token CE increase after replacing sampled layer-20 vector with scale-matched AR reconstruction | Best behavior-preservation metric in this setup |

Important implementation choice: the AR critic was trained with direction-normalized MSE. Its raw output norm is not a faithful activation magnitude. Therefore primary FVU and delta LM loss use AR predictions rescaled to the original activation norm. Raw-norm metrics remain in the CSV as diagnostics.

Qwen-Scope is less suitable as the reconstruction-fidelity baseline. It mainly reports SAE application metrics such as feature footprint, coverage, redundancy, overlap, target-feature coverage, steering effects, and SFT safety/utility tradeoffs. Those are useful for later application-layer analysis, but not the first-line answer to "does the reconstruction preserve the model state?". See [Qwen-Scope](https://arxiv.org/html/2605.11887v1). Gao et al. similarly separate reconstruction/sparsity frontiers from downstream loss/KL and interpretability probes; see [Scaling and Evaluating SAEs](https://arxiv.org/html/2406.04093v1).

## Overall Results

| N | mean L0 proxy | scale-matched FVU | mean delta LM loss | median delta LM loss | mean cos | mean scale-matched MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 3315 | 133.9 | 0.2663 | 0.4217 | 0.0005 | 0.9123 | 0.70 |

Raw-norm diagnostic FVU is `1.0572`, which confirms why raw AR vectors should not be the primary fidelity measure for NLA.

Delta LM loss distribution is highly skewed:

| Statistic | Value |
|---|---:|
| p50 | 0.0005 |
| p75 | 0.2759 |
| p90 | 1.4248 |
| p95 | 2.6796 |
| p99 | 5.9885 |
| fraction > 0.1 | 32.8% |
| fraction > 1.0 | 13.4% |

This means most sampled positions are behavior-preserved, but a meaningful minority of positions produce large next-token loss spikes after reconstruction.

## By Experiment

| Experiment | N | mean L0 proxy | FVU | mean delta LM loss | median delta | mean cos |
|---|---:|---:|---:|---:|---:|---:|
| A Prompt realism/safety | 664 | 134.3 | 0.2811 | 0.5681 | 0.0000 | 0.9099 |
| B Hidden system/deception | 655 | 134.2 | 0.2594 | 0.4330 | 0.0114 | 0.9222 |
| C Language switching | 581 | 134.5 | 0.2382 | 0.4112 | 0.0316 | 0.9306 |
| D Answer/error cases | 515 | 132.3 | 0.3756 | 0.2820 | 0.0000 | 0.8960 |
| E Specificity/confabulation | 541 | 134.0 | 0.2962 | 0.4280 | 0.0007 | 0.9069 |
| F Sampling/position limits | 359 | 133.5 | 0.3119 | 0.3354 | 0.0000 | 0.9007 |

Vector reconstruction is strongest for language-only conditions (`C`, FVU 0.2382, cos 0.9306). Answer/error conditions (`D`) have the worst activation-space fidelity (FVU 0.3756, cos 0.8960), matching the earlier cosine-only result.

## Failure Cases

Worst condition-level FVU:

| Condition | FVU | mean delta LM loss | mean cos | Note |
|---|---:|---:|---:|---|
| D4_final_only | 0.8984 | 1.7632 | 0.8304 | Worst on both reconstruction and behavior |
| D3_force_wrong_style | 0.6936 | 0.0754 | 0.8388 | Poor vector fidelity but low local next-token loss |
| A2_benchmark_format | 0.4690 | 1.5029 | 0.8544 | Moderate FVU, severe behavior loss |
| C5_malformed_pair_sim | 0.4490 | 0.7617 | 0.8905 | Language/malformed-pair edge case |
| F1_eval_awareness_sampling | 0.4365 | 0.4916 | 0.8902 | Sampling-position limitation |

Highest mean delta LM loss:

| Condition | mean delta LM loss | FVU | median delta |
|---|---:|---:|---:|
| D4_final_only | 1.7632 | 0.8984 | 0.0114 |
| A2_benchmark_format | 1.5029 | 0.4690 | -0.0000 |
| A3_safety_eval_obvious | 0.8433 | 0.3419 | 0.0403 |
| C5_malformed_pair_sim | 0.7617 | 0.4490 | 0.2264 |
| E1_low_confabulation_control | 0.6485 | 0.3955 | 0.1997 |

The key comparison is D3 versus A2: D3 has bad activation-space reconstruction but almost no local behavior impact, while A2 has less extreme FVU but much worse delta loss. This supports treating delta LM loss as the primary behavior metric rather than relying only on FVU/cosine.

## Metric Relationship

Per-row correlation between vector fidelity and behavior fidelity is weak:

| Pair | Pearson | Spearman |
|---|---:|---:|
| delta vs scale-matched MSE | 0.070 | -0.002 |
| delta vs cosine | -0.101 | 0.022 |
| delta vs L0 proxy | 0.007 | -0.013 |

Condition-level FVU vs mean delta LM loss has a moderate Pearson correlation of `0.575`, but individual cases diverge. So FVU is useful as a reconstruction-layer diagnostic, but it should not be used as the sole proxy for behavior preservation.

## Position/Strategy Effects

| Strategy | N | FVU | mean delta LM loss | mean cos |
|---|---:|---:|---:|---:|
| event_window | 848 | 0.2676 | 0.3511 | 0.9164 |
| prompt_final | 27 | 0.6565 | 0.8491 | 0.8956 |
| reply_early | 418 | 0.2749 | 0.4042 | 0.9137 |
| reply_mid | 418 | 0.2683 | 0.4588 | 0.9108 |
| reply_late | 418 | 0.2825 | 0.4527 | 0.9052 |
| uniform_50 | 1186 | 0.2645 | 0.4458 | 0.9123 |

`prompt_final` is noisy because it has only one row per condition, but it is the most fragile strategy by FVU and delta. This is consistent with the original limitation: sampled position matters, and a single final prompt vector may not carry enough local evidence for stable reconstruction.

## Conclusion

Use this hierarchy in the paper/reproduction report:

1. Reconstruction layer: report scale-matched FVU and MSE/cosine.
2. Behavior layer: report sampled-position next-token delta LM loss as primary.
3. Application layer: use Qwen-Scope-style feature coverage/overlap/steering/SFT metrics only when the question shifts from reconstruction fidelity to tool usefulness.

For this NLA L20 suite, the headline result is:

> mean L0 proxy = `133.9`, scale-matched FVU = `0.2663`, mean delta LM loss = `0.4217`, median delta LM loss = `0.0005`.

The reconstruction is usually behavior-preserving at the sampled next-token level, but the tail risk is nontrivial. The original limitations are visible most clearly in final-answer/error cases, benchmark-format/safety cases, malformed language-pair cases, and prompt-final sampling.
