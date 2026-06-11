#!/usr/bin/env bash
# Shared local paths for the Qwen NLA reproduction setup.
# Source this file before launching SGLang or running smoke tests.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export QWEN_BASE_MODEL="${QWEN_BASE_MODEL:-/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct}"
export QWEN_NLA_AV="${QWEN_NLA_AV:-/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av}"
export QWEN_NLA_AR="${QWEN_NLA_AR:-/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.0}"
export FLASHINFER_NVCC="${FLASHINFER_NVCC:-$CUDA_HOME/bin/nvcc}"

case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
esac

case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

export NLA_TMP_ROOT="${NLA_TMP_ROOT:-/NAS/chennc/NashChennc/.tmp}"
export TMPDIR="${TMPDIR:-$NLA_TMP_ROOT/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NLA_TMP_ROOT/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NLA_TMP_ROOT/xdg-cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$NLA_TMP_ROOT/flashinfer}"

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE"

_nla_local_no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${_nla_local_no_proxy}"
export no_proxy="${no_proxy:+$no_proxy,}${_nla_local_no_proxy}"
unset _nla_local_no_proxy
