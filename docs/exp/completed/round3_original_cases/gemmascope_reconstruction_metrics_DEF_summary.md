# Gemma Scope-style Reconstruction Metrics for Original Cases L20

- Base model: `/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AR critic/reconstructor: `/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- Layer patched for delta LM loss: `20`
- Rows: `1415`

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
| overall | 1415 | 133.2 | 0.3042 | 0.3517 | 0.0001 | 0.9014 | 0.77 |

- Raw-norm diagnostic overall FVU: `1.1334`

## By Experiment

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 515 | 132.3 | 0.3756 | 0.2820 | 0.0000 | 0.8960 | 0.78 |
| E | 541 | 134.0 | 0.2962 | 0.4280 | 0.0007 | 0.9069 | 0.76 |
| F | 359 | 133.5 | 0.3119 | 0.3354 | 0.0000 | 0.9007 | 0.77 |

## By Condition

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
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
| event_window | 384 | 134.1 | 0.2978 | 0.3405 | 0.0001 | 0.9066 | 0.74 |
| prompt_final | 12 | 132.9 | 0.6677 | 0.7160 | 0.0133 | 0.8945 | 0.72 |
| reply_early | 178 | 133.7 | 0.3005 | 0.3466 | 0.0001 | 0.9064 | 0.74 |
| reply_late | 178 | 130.3 | 0.3444 | 0.3740 | 0.0003 | 0.8900 | 0.82 |
| reply_mid | 178 | 134.1 | 0.3114 | 0.3587 | 0.0001 | 0.8991 | 0.78 |
| uniform_50 | 485 | 133.1 | 0.3053 | 0.3430 | 0.0000 | 0.9006 | 0.78 |
