#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/qwen_nla_env.sh"

PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-flashinfer}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES is not set; SGLang will use its default GPU visibility." >&2
fi

args=(
    python -m sglang.launch_server
    --model-path "$QWEN_NLA_AV" \
    --port "$PORT" \
    --disable-radix-cache \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --attention-backend "$ATTENTION_BACKEND" \
    --sampling-backend "$SAMPLING_BACKEND" \
    --trust-remote-code
)

if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    args+=(--disable-cuda-graph)
fi

if [[ "$SKIP_SERVER_WARMUP" == "1" ]]; then
    args+=(--skip-server-warmup)
fi

if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=($SGLANG_EXTRA_ARGS)
    args+=("${extra_args[@]}")
fi

"${args[@]}"
