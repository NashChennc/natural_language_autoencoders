# Qwen NLA 复现实验简报

日期：2026-05-20  
仓库：`/NAS/chennc/NashChennc/natural_language_autoencoders`  
GPU：GPU4，NVIDIA A40

## 结论

Qwen2.5-7B layer-20 NLA inference 已在本地跑通。正式运行使用真实 base Qwen activation，不是随机 smoke vector。

高性能后端已恢复并验证：

- SGLang `attention_backend=flashinfer`
- SGLang `sampling_backend=flashinfer`
- CUDA graph 开启
- `nvcc` 固定为 `/usr/local/cuda-12.0/bin/nvcc`

正式解码结果与上下文一致：base reply 以 “As Qwen, created by Alibaba Cloud...” 开头，NLA 对前 8 个 token 的解释集中在 Qwen 身份、Alibaba Cloud 归属、AI assistant 能力说明等语义上。

## 模型路径

```text
Base: /NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct
AV:   /NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av
AR:   /NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar
```

关键参数：

```text
d_model=3584
layer_index=20
injection_scale=150.0
injection_token='㈎' id=149705
```

## 产物

```text
/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/qwen_reply_layer20.parquet
/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/qwen_reply_layer20.parquet.manifest.json
/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/nla_decode_temp0.log
```

## 关键命令

生成真实 activation parquet：

```bash
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
source scripts/qwen_nla_env.sh

CUDA_VISIBLE_DEVICES=4 \
python scripts/make_qwen_layer20_demo_parquet.py \
  --base-model "$QWEN_BASE_MODEL" \
  --output "$NLA_TMP_ROOT/qwen_nla_formal/qwen_reply_layer20.parquet" \
  --max-rows 16
```

启动 AV 服务：

```bash
CUDA_VISIBLE_DEVICES=4 PORT=30000 bash scripts/launch_qwen_nla_sglang.sh
```

正式解码：

```bash
CUDA_VISIBLE_DEVICES=4 \
python nla_inference.py "$QWEN_NLA_AV" \
  --sglang-url http://localhost:30000 \
  --parquet "$NLA_TMP_ROOT/qwen_nla_formal/qwen_reply_layer20.parquet" \
  --n 8 \
  --temperature 0 \
  --max-new-tokens 240
```

## 本地适配

- `scripts/qwen_nla_env.sh`：统一模型路径、hf-mirror、CUDA 12.0、FlashInfer JIT cache、localhost no-proxy。
- `scripts/launch_qwen_nla_sglang.sh`：默认启动 FlashInfer + CUDA graph。
- `scripts/make_qwen_layer20_demo_parquet.py`：生成正式 demo parquet。
- `nla_inference.py`：默认 `httpx trust_env=False`，避免 localhost 请求被 SOCKS proxy 环境变量影响。

## 当前状态

实验成功，SGLang 服务已停止，GPU4 已释放。
