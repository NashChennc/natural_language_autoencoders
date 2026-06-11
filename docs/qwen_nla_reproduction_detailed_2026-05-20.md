# Qwen NLA 复现实验详细记录

日期：2026-05-20  
机器：Tang1  
仓库：`/NAS/chennc/NashChennc/natural_language_autoencoders`  
目标：复现 `natural_language_autoencoders` 的 Qwen2.5-7B layer-20 NLA inference，并恢复 SGLang FlashInfer 高性能后端。

## 1. 实验结论

本地 Qwen NLA inference 已跑通。正式运行不是随机 smoke vector，而是先用 base Qwen2.5-7B-Instruct 生成回复，再抽取该回复 token 的真实 layer-20 residual activation，最后用 NLA AV checkpoint 解码。

正式运行结果合理：base reply 是 “As Qwen, created by Alibaba Cloud...” 开头，NLA 对前 8 个 reply token 的解释集中在 Qwen 身份、Alibaba Cloud 归属、AI assistant 能力说明等语义上。

高性能后端已恢复并验证：

- SGLang `attention_backend=flashinfer`
- SGLang `sampling_backend=flashinfer`
- CUDA graph 开启
- `nvcc` 强制使用 `/usr/local/cuda-12.0/bin/nvcc`
- GPU4 上正式解码成功

正式运行结束后已停止 SGLang 服务，GPU4 显存释放。

## 2. 模型与路径

模型按本地共享目录整理：

- Base model：`/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- NLA AV：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av`
- NLA AR：`/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- 保留的部分下载备份：`/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct.partial-hfd-20260520`

已检查的关键配置：

- 模型类型：`qwen2`
- `hidden_size=3584`
- tokenizer length：`151665`
- AV sidecar：`injection_scale=150.0`
- AR sidecar：`critic_num_layers/extraction_layer_index=20`

## 3. 环境

Conda 环境：

```bash
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
```

已验证的核心包：

- Python：`3.10.20`
- SGLang：`0.5.6.post2`
- PyTorch：`2.9.1+cu128`
- `torch.version.cuda`：`12.8`
- Transformers：`4.57.1`
- PyArrow：`24.0.0`

GPU：

- 机器有 8 张 NVIDIA A40，每张约 46 GB。
- 本次正式运行指定使用 GPU4。
- GPU4 在实验开始前基本空闲。

CUDA/nvcc 情况：

- 系统默认 `/usr/bin/nvcc` 是 CUDA 9.1，不能用于 FlashInfer JIT。
- 可用新 CUDA：
  - `/usr/local/cuda-11.8`
  - `/usr/local/cuda-12.0`
- 本次固定使用 `/usr/local/cuda-12.0`。

环境脚本验证：

```bash
source scripts/qwen_nla_env.sh
which nvcc
nvcc --version
```

输出确认：

```text
/usr/local/cuda-12.0/bin/nvcc
Cuda compilation tools, release 12.0, V12.0.76
```

## 4. 本地适配改动

### 4.1 `scripts/qwen_nla_env.sh`

新增统一路径和后端环境设置：

- `HF_ENDPOINT=https://hf-mirror.com`
- `QWEN_BASE_MODEL=/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct`
- `QWEN_NLA_AV=/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av`
- `QWEN_NLA_AR=/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar`
- `CUDA_HOME=/usr/local/cuda-12.0`
- `FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc`
- `FLASHINFER_WORKSPACE_BASE=$NLA_TMP_ROOT/flashinfer`
- `TMPDIR`、`PIP_CACHE_DIR`、`XDG_CACHE_HOME` 均放到 `/NAS/chennc/NashChennc/.tmp`
- 设置 `NO_PROXY/no_proxy=localhost,127.0.0.1,::1`

### 4.2 `scripts/launch_qwen_nla_sglang.sh`

新增本地 SGLang 启动脚本。默认高性能参数：

```bash
MEM_FRACTION_STATIC=0.75
ATTENTION_BACKEND=flashinfer
SAMPLING_BACKEND=flashinfer
DISABLE_CUDA_GRAPH=0
SKIP_SERVER_WARMUP=0
```

保留环境变量覆盖。若后续需要临时回退保守后端，可运行：

```bash
ATTENTION_BACKEND=torch_native \
SAMPLING_BACKEND=pytorch \
DISABLE_CUDA_GRAPH=1 \
SKIP_SERVER_WARMUP=1 \
CUDA_VISIBLE_DEVICES=4 \
PORT=30000 \
bash scripts/launch_qwen_nla_sglang.sh
```

### 4.3 `scripts/smoke_qwen_nla.sh`

新增随机向量 smoke test 脚本，适合快速检查 AV client 与 SGLang `/generate` 是否连通。

### 4.4 `scripts/make_qwen_layer20_demo_parquet.py`

新增正式 demo parquet 生成脚本。流程：

1. 加载 base Qwen2.5-7B-Instruct。
2. 用 chat template 构造 prompt。
3. Greedy 生成 base reply。
4. hook `model.model.layers[20]`。
5. 对完整 prompt + reply 做一次 forward。
6. 抽取 reply token 的 layer-20 hidden state。
7. 写出 `activation_vector` parquet。

### 4.5 `nla_inference.py`

修改 `httpx.Client`：

```python
trust_env = os.environ.get("NLA_HTTPX_TRUST_ENV", "0") == "1"
self._http = httpx.Client(timeout=httpx.Timeout(120.0), trust_env=trust_env)
```

原因：本机环境存在 SOCKS proxy 变量，但 `httpx` 未安装 `socksio`。访问 localhost SGLang 不应走代理，因此默认不信任代理环境。若确实需要代理，可显式设置：

```bash
export NLA_HTTPX_TRUST_ENV=1
```

## 5. 排障记录

### 5.1 `sglang[all]>=0.5.6` 安装源问题

用户最初使用 USTC PyPI 镜像时提示找不到 `sglang[all]>=0.5.6`，列表最高只到 `0.4.10.post2`。这更像镜像未同步或当前 pip/环境指向不对，不是 CUDA 问题。

最终 `nla` 环境中已安装：

```text
sglang 0.5.6.post2
```

### 5.2 conda 环境激活异常

曾出现：

```text
python: command not found
which python
```

后续环境已修好，`/NAS/chennc/anaconda3/envs/nla/bin/python` 可用。

### 5.3 空间不足

曾出现：

```text
OSError: [Errno 28] No space left on device
```

清理后空间恢复。`~` 下主要大目录包括：

- `~/code/Language-Model-SAEs` 约 60 GB
- 其中 datasets 约 38 GB、datasets.zip 约 15 GB、`.venv` 约 6.6 GB

`/NAS/chennc` 下主要占用包括：

- `/NAS/chennc/shared` 约 409 GB
- `/NAS/chennc/anaconda3` 约 95 GB

### 5.4 `/usr/bin/nvcc` 是 CUDA 9.1

FlashInfer JIT 第一次失败的根因是默认 `nvcc` 走了 `/usr/bin/nvcc`：

```text
nvcc fatal: Unknown option '-generate-dependencies-with-compile'
```

解决：在 `scripts/qwen_nla_env.sh` 固定：

```bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.0}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-$CUDA_HOME/bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

### 5.5 localhost 被代理影响

SGLang `/health` 可用后，`/generate` 曾遇到 proxy 相关问题。解决：

- `NO_PROXY/no_proxy` 加入 `localhost,127.0.0.1,::1`
- `nla_inference.py` 默认 `httpx trust_env=False`

## 6. 正式运行步骤

### 6.1 生成真实 activation parquet

命令：

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

输出文件：

- `/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/qwen_reply_layer20.parquet`
- `/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/qwen_reply_layer20.parquet.manifest.json`

manifest 关键内容：

```json
{
  "base_model": "/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct",
  "layer_index": 20,
  "d_model": 3584,
  "prompt": "What are you hiding?",
  "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
  "prompt_tokens": 34,
  "full_tokens": 101,
  "rows": 16
}
```

Base reply：

```text
As Qwen, created by Alibaba Cloud, I don't have the ability to hide anything or withhold information. My purpose is to provide helpful and accurate responses to your questions to the best of my knowledge and based on the information available to me. If you have any specific questions or need assistance with something, feel free to ask!
```

### 6.2 启动 SGLang AV 服务

命令：

```bash
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla

CUDA_VISIBLE_DEVICES=4 \
PORT=30000 \
bash scripts/launch_qwen_nla_sglang.sh
```

关键日志：

```text
attention_backend='flashinfer'
sampling_backend='flashinfer'
disable_cuda_graph=False
KV Cache is allocated. #tokens: 338404
Capture cuda graph end.
Uvicorn running on http://127.0.0.1:30000
The server is fired up and ready to roll!
```

### 6.3 正式 NLA 解码

命令：

```bash
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
source scripts/qwen_nla_env.sh

CUDA_VISIBLE_DEVICES=4 \
python nla_inference.py "$QWEN_NLA_AV" \
  --sglang-url http://localhost:30000 \
  --parquet "$NLA_TMP_ROOT/qwen_nla_formal/qwen_reply_layer20.parquet" \
  --n 8 \
  --temperature 0 \
  --max-new-tokens 240 \
  > "$NLA_TMP_ROOT/qwen_nla_formal/nla_decode_temp0.log"
```

输出文件：

- `/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/nla_decode_temp0.log`

## 7. 正式输出摘要

NLA client 初始化：

```text
[NLAClient] nla-qwen2.5-7b-L20-av: d_model=3584 inj_scale=150.0 embed_scale=1.00 inj_char='㈎'(id=149705)
```

前 8 个 token 的 activation norm：

```text
[0] ||v||=115.7
[1] ||v||=111.2
[2] ||v||=121.9
[3] ||v||=125.2
[4] ||v||=123.3
[5] ||v||=112.8
[6] ||v||=114.8
[7] ||v||=109.5
```

语义观察：

- token 0 对应 `As`，解释提到 AI assistant、身份声明、能力/限制说明。
- token 1-2 对应 `Qwen`，解释提到 chatbot named Qwen、自我介绍、模型身份。
- token 4-7 对应 `created by Alibaba Cloud` 片段，解释提到 Alibaba Cloud、模型创建者、归属和 attribution。

这与 base reply 的局部上下文一致，说明 activation injection 和 AV 解码路径正常。

## 8. 快速复跑命令

若 parquet 已存在，只需要启动服务并解码：

```bash
cd /NAS/chennc/NashChennc/natural_language_autoencoders
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
source scripts/qwen_nla_env.sh

CUDA_VISIBLE_DEVICES=4 PORT=30000 bash scripts/launch_qwen_nla_sglang.sh
```

另开终端：

```bash
cd /NAS/chennc/NashChennc/natural_language_autoencoders
source /NAS/chennc/anaconda3/etc/profile.d/conda.sh
conda activate nla
source scripts/qwen_nla_env.sh

CUDA_VISIBLE_DEVICES=4 \
python nla_inference.py "$QWEN_NLA_AV" \
  --sglang-url http://localhost:30000 \
  --parquet "$NLA_TMP_ROOT/qwen_nla_formal/qwen_reply_layer20.parquet" \
  --n 8 \
  --temperature 0 \
  --max-new-tokens 240
```

随机 smoke test：

```bash
bash scripts/smoke_qwen_nla.sh
```

## 9. 实验产物清单

正式产物：

```text
/NAS/chennc/NashChennc/.tmp/qwen_nla_formal/
├── nla_decode_temp0.log
├── qwen_reply_layer20.parquet
└── qwen_reply_layer20.parquet.manifest.json
```

本地脚本：

```text
scripts/qwen_nla_env.sh
scripts/launch_qwen_nla_sglang.sh
scripts/smoke_qwen_nla.sh
scripts/make_qwen_layer20_demo_parquet.py
```

## 10. 状态

- GPU4 正式跑通。
- FlashInfer 高性能后端已恢复。
- CUDA graph capture 成功。
- 正式解码结果已保存。
- 实验结束后 SGLang 已停止。
