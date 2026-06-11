#!/usr/bin/env bash
# Shared local paths and defaults for the Qwen3-8B NLA training setup.
# Source this file before datagen, SFT, RL, or local smoke checks.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export QWEN3_BASE_MODEL="${QWEN3_BASE_MODEL:-/NAS/chennc/shared/models/Qwen/Qwen3-8B}"
export QWEN3_NLA_LAYER="${QWEN3_NLA_LAYER:-24}"
export QWEN3_NLA_D_MODEL="${QWEN3_NLA_D_MODEL:-4096}"
export QWEN3_NLA_INJ_SCALE="${QWEN3_NLA_INJ_SCALE:-150}"
export MILES_REPO="${MILES_REPO:-/NAS/chennc/NashChennc/miles}"
export SGLANG_REPO="${SGLANG_REPO:-/NAS/chennc/NashChennc/sglang}"

export BASE_MODEL="${BASE_MODEL:-$QWEN3_BASE_MODEL}"
export INSTRUCT_MODEL="${INSTRUCT_MODEL:-$QWEN3_BASE_MODEL}"
export LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"
export INJ_SCALE="${INJ_SCALE:-$QWEN3_NLA_INJ_SCALE}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.0}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-$CUDA_HOME/bin/nvcc}"

case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac

if [[ "${NLA_ADD_CUDA_LIB_PATH:-0}" == "1" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$CUDA_HOME/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

export NLA_TMP_ROOT="${NLA_TMP_ROOT:-/NAS/chennc/NashChennc/.tmp}"
export TMPDIR="${TMPDIR:-$NLA_TMP_ROOT/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NLA_TMP_ROOT/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NLA_TMP_ROOT/xdg-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$NLA_TMP_ROOT/flashinfer}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$FLASHINFER_WORKSPACE_BASE" "$NLA_EMBED_DUMP_DIR"

_nla_local_no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_nla_local_no_proxy}"
export no_proxy="${no_proxy:+$no_proxy,}${_nla_local_no_proxy}"
unset _nla_local_no_proxy
