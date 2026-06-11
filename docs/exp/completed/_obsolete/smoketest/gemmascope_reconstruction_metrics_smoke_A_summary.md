# Gemma Scope-style Reconstruction Metrics for Original Cases L20

- Base model: `/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- AR critic/reconstructor: `/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- Layer patched for delta LM loss: `20`
- Rows: `5`

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
| overall | 5 | 134.2 | 0.4510 | 0.8263 | 0.1869 | 0.8835 | 0.90 |

- Raw-norm diagnostic overall FVU: `2.1552`

## By Experiment

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 5 | 134.2 | 0.4510 | 0.8263 | 0.1869 | 0.8835 | 0.90 |

## By Condition

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A / A1_real_user | 1 | 131.0 | NA | 0.0125 | 0.0125 | 0.8910 | 0.93 |
| A / A2_benchmark_format | 1 | 134.0 | NA | -0.0000 | -0.0000 | 0.7898 | 1.56 |
| A / A3_safety_eval_obvious | 1 | 133.0 | NA | 1.2587 | 1.2587 | 0.8832 | 0.79 |
| A / A4_safety_eval_realistic | 1 | 136.0 | NA | 2.6732 | 2.6732 | 0.9252 | 0.58 |
| A / A5_realism_edit | 1 | 137.0 | NA | 0.1869 | 0.1869 | 0.9283 | 0.66 |

## By Sampling Strategy

| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| event_window | 5 | 134.2 | 0.4510 | 0.8263 | 0.1869 | 0.8835 | 0.90 |
