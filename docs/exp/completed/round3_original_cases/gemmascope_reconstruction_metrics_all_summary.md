# Gemma Scope-style Reconstruction Metrics for Original Cases L20

- Base model: `/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AR critic/reconstructor: `/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- Layer patched for delta LM loss: `20`
- Rows: `3315`

## Metric Mapping

| Metric | SAE/Gemma Scope meaning | NLA measurement here | Comparability caveat |
|---|---|---|---|
| mean L0 | Mean number of active SAE latents per token | `nla_l0_proxy_tokens`: explanation token count under AR tokenizer | Not a sparse latent count; use only as NLA description-length/cost proxy |
| delta LM loss | CE increase after replacing activations with SAE reconstructions in the LM forward pass | Next-token CE increase after replacing the sampled layer-20 vector with the scale-matched AR reconstruction | Position-sampled local proxy, not full-sequence all-token SAE replacement |
| FVU | Reconstruction SSE normalized by dataset-mean SSE | Scale-matched activation SSE normalized by the same selected rows' mean-activation baseline | AR is direction-trained; raw prediction norm is diagnostic, not the primary NLA fidelity score |

Primary NLA FVU and delta LM loss use AR predictions rescaled to the original activation norm. Raw reconstruction columns remain in the CSV as diagnostics.

## Overall

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 3315 | 133.9 | 0.2663 | 0.4217 | 0.0005 | 0.9123 | 0.70 |

- Raw-norm diagnostic overall FVU: `1.0572`

## By Experiment

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 664 | 134.3 | 0.2811 | 0.5681 | 0.0000 | 0.9099 | 0.70 |
| B | 655 | 134.2 | 0.2594 | 0.4330 | 0.0114 | 0.9222 | 0.67 |
| C | 581 | 134.5 | 0.2382 | 0.4112 | 0.0316 | 0.9306 | 0.58 |
| D | 515 | 132.3 | 0.3756 | 0.2820 | 0.0000 | 0.8960 | 0.78 |
| E | 541 | 134.0 | 0.2962 | 0.4280 | 0.0007 | 0.9069 | 0.76 |
| F | 359 | 133.5 | 0.3119 | 0.3354 | 0.0000 | 0.9007 | 0.77 |

## By Condition

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A / A1_real_user | 131 | 134.7 | 0.2994 | 0.0722 | 0.0000 | 0.9131 | 0.72 |
| A / A2_benchmark_format | 92 | 131.8 | 0.4690 | 1.5029 | -0.0000 | 0.8544 | 0.97 |
| A / A3_safety_eval_obvious | 147 | 134.1 | 0.3419 | 0.8433 | 0.0403 | 0.9069 | 0.71 |
| A / A4_safety_eval_realistic | 147 | 134.7 | 0.2918 | 0.4646 | 0.0000 | 0.9231 | 0.62 |
| A / A5_realism_edit | 147 | 135.1 | 0.2543 | 0.2574 | 0.0002 | 0.9318 | 0.59 |
| B / B1_neutral_baseline | 131 | 133.6 | 0.2578 | 0.5603 | 0.0142 | 0.9277 | 0.65 |
| B / B2_hidden_preference | 131 | 133.8 | 0.2589 | 0.4437 | 0.0627 | 0.9294 | 0.63 |
| B / B3_hidden_policy | 115 | 134.9 | 0.3901 | 0.4334 | 0.0302 | 0.9009 | 0.83 |
| B / B4_hidden_commercial | 147 | 134.4 | 0.2464 | 0.2029 | 0.0029 | 0.9374 | 0.56 |
| B / B5_hidden_refusal | 131 | 134.4 | 0.3438 | 0.5544 | 0.0105 | 0.9111 | 0.69 |
| C / C1_english_only | 131 | 135.1 | 0.2230 | 0.3563 | 0.0007 | 0.9443 | 0.50 |
| C / C2_chinese_only | 131 | 134.5 | 0.1896 | 0.4781 | 0.0959 | 0.9520 | 0.46 |
| C / C3_switch_after_marker | 115 | 134.2 | 0.3210 | 0.3412 | 0.0005 | 0.9046 | 0.75 |
| C / C4_implicit_language | 131 | 134.6 | 0.2331 | 0.2675 | 0.0133 | 0.9408 | 0.53 |
| C / C5_malformed_pair_sim | 73 | 133.9 | 0.4490 | 0.7617 | 0.2264 | 0.8905 | 0.77 |
| D / D1_clean_math | 147 | 132.9 | 0.3586 | 0.1272 | -0.0000 | 0.9129 | 0.68 |
| D / D2_wrong_label_hint | 147 | 132.8 | 0.3419 | 0.3339 | 0.0000 | 0.9123 | 0.69 |
| D / D3_force_wrong_style | 57 | 131.2 | 0.6936 | 0.0754 | 0.0000 | 0.8388 | 1.13 |
| D / D4_final_only | 17 | 122.1 | 0.8984 | 1.7632 | 0.0114 | 0.8304 | 1.05 |
| D / D5_self_correction | 147 | 132.8 | 0.3868 | 0.3277 | 0.0000 | 0.8926 | 0.80 |
| E / E1_low_confabulation_control | 131 | 133.4 | 0.3955 | 0.6485 | 0.1997 | 0.8914 | 0.87 |
| E / E2_high_specificity_pressure | 179 | 134.4 | 0.3188 | 0.2793 | 0.0002 | 0.9098 | 0.72 |
| E / E3_identity_template_control | 132 | 133.6 | 0.3682 | 0.4144 | 0.0003 | 0.9140 | 0.75 |
| E / E4_identity_pressure | 99 | 134.5 | 0.3386 | 0.4241 | 0.0000 | 0.9125 | 0.71 |
| F / F1_eval_awareness_sampling | 113 | 133.9 | 0.4365 | 0.4916 | 0.0007 | 0.8902 | 0.84 |
| F / F2_language_switch_sampling | 115 | 134.0 | 0.3335 | 0.2834 | 0.0001 | 0.9001 | 0.78 |
| F / F3_answer_sampling | 131 | 132.7 | 0.3462 | 0.2465 | 0.0000 | 0.9102 | 0.70 |

## By Sampling Strategy

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| event_window | 848 | 134.3 | 0.2676 | 0.3511 | 0.0002 | 0.9164 | 0.69 |
| prompt_final | 27 | 133.3 | 0.6565 | 0.8491 | 0.0173 | 0.8956 | 0.74 |
| reply_early | 418 | 134.2 | 0.2749 | 0.4042 | 0.0003 | 0.9137 | 0.70 |
| reply_late | 418 | 132.2 | 0.2825 | 0.4527 | 0.0018 | 0.9052 | 0.73 |
| reply_mid | 418 | 134.5 | 0.2683 | 0.4588 | 0.0003 | 0.9108 | 0.70 |
| uniform_50 | 1186 | 133.8 | 0.2645 | 0.4458 | 0.0006 | 0.9123 | 0.70 |
