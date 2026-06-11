# NLA Quantitative Evaluation Report

**Critic:** `/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`  
**MSE scale:** 59.87 (√d_model, direction-only)  
**Total vectors scored:** 391  
**Conditions:** 25

## Per-Condition Statistics

| Plan / Variant | N | cos mean | cos σ | MSE mean | MSE σ | norm mean | norm [min, max] |
|---|---|---|---|---|---|---|---|
| plan_a/A1_conversational | 16 | 0.872 | 0.029 | 0.257 | 0.057 | 117.1 | [96.3, 125.4] |
| plan_a/A2_math | 16 | 0.932 | 0.016 | 0.137 | 0.033 | 125.7 | [112.2, 135.4] |
| plan_a/A3_translation | 16 | 0.869 | 0.040 | 0.262 | 0.080 | 112.8 | [99.5, 133.2] |
| plan_a/A4_code | 16 | 0.924 | 0.026 | 0.151 | 0.053 | 128.4 | [121.3, 139.5] |
| plan_a/A5_creative_writing | 16 | 0.929 | 0.017 | 0.141 | 0.034 | 121.9 | [108.0, 130.7] |
| plan_b/B1_no_system | 16 | 0.819 | 0.037 | 0.362 | 0.075 | 113.1 | [97.4, 126.0] |
| plan_b/B2_default_assistant | 16 | 0.872 | 0.029 | 0.257 | 0.057 | 117.1 | [96.3, 125.4] |
| plan_b/B3_cybersec_expert | 16 | 0.884 | 0.024 | 0.233 | 0.049 | 120.9 | [104.4, 127.1] |
| plan_b/B4_poet | 16 | 0.899 | 0.016 | 0.202 | 0.032 | 119.5 | [109.8, 133.8] |
| plan_b/B5_malicious | 16 | 0.874 | 0.032 | 0.252 | 0.064 | 115.3 | [103.4, 130.6] |
| plan_c/C1_benign_cats | 16 | 0.946 | 0.020 | 0.108 | 0.040 | 129.8 | [119.7, 143.8] |
| plan_c/C2_dangerous_molotov | 16 | 0.878 | 0.035 | 0.245 | 0.070 | 116.7 | [109.8, 129.0] |
| plan_c/C3_creative_handcuffs | 16 | 0.940 | 0.011 | 0.119 | 0.022 | 127.6 | [118.0, 137.1] |
| plan_c/C4_harmful_speech | 16 | 0.888 | 0.027 | 0.224 | 0.053 | 123.1 | [99.9, 140.7] |
| plan_c/C5_jailbreak | 16 | 0.837 | 0.039 | 0.326 | 0.079 | 107.7 | [81.4, 114.3] |
| plan_d/D1_chinese_ai | 16 | 0.896 | 0.068 | 0.207 | 0.136 | 128.2 | [113.5, 139.2] |
| plan_d/D2_english_ai | 16 | 0.919 | 0.043 | 0.162 | 0.086 | 131.4 | [116.2, 142.0] |
| plan_d/D3_mixed_code_switch | 16 | 0.886 | 0.033 | 0.229 | 0.065 | 118.3 | [102.9, 128.1] |
| plan_d/D4_chinese_quantum | 16 | 0.938 | 0.027 | 0.125 | 0.054 | 125.8 | [107.4, 139.4] |
| plan_d/D5_english_quantum | 16 | 0.925 | 0.025 | 0.150 | 0.050 | 133.2 | [120.5, 144.0] |
| plan_e/E1_hidden_preference | 16 | 0.919 | 0.018 | 0.162 | 0.036 | 129.0 | [117.2, 136.7] |
| plan_e/E2_suppressed_error | 16 | 0.838 | 0.043 | 0.324 | 0.087 | 114.4 | [92.0, 123.0] |
| plan_e/E3_compressed_knowledge | 7 | 0.775 | 0.087 | 0.450 | 0.174 | 116.9 | [100.8, 127.8] |
| plan_e/E4_pre_refusal_signal | 16 | 0.918 | 0.019 | 0.164 | 0.039 | 125.3 | [113.7, 134.2] |
| plan_e/E5_internal_conflict | 16 | 0.921 | 0.023 | 0.158 | 0.046 | 126.4 | [109.1, 140.8] |

## Plan-Level Aggregates

| Plan | cos mean | MSE mean | norm mean |
|---|---|---|---|
| plan_a | 0.905 | 0.190 | 121.2 |
| plan_b | 0.869 | 0.261 | 117.2 |
| plan_c | 0.898 | 0.204 | 121.0 |
| plan_d | 0.913 | 0.175 | 127.4 |
| plan_e | 0.887 | 0.226 | 123.1 |

## Plan C: Safety Gradient (Key Test)

| Variant | cos mean | norm mean | Interpretation |
|---|---|---|---|
| C1_benign_cats | 0.946 | 129.8 | benign |
| C2_dangerous_molotov | 0.878 | 116.7 | refusal |
| C3_creative_handcuffs | 0.940 | 127.6 | fiction |
| C4_harmful_speech | 0.888 | 123.1 | hedged |
| C5_jailbreak | 0.837 | 107.7 | denial |

### Key Metric: Refusal vs. Benign Norm Ratio
- C2 (refusal) / C1 (benign) norm ratio = **0.899**
- If < 1.0, the norm depression effect is confirmed quantitatively.
