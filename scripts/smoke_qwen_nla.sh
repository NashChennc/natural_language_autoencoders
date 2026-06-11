#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/qwen_nla_env.sh"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"

python "$REPO_ROOT/nla_inference.py" "$QWEN_NLA_AV" \
    --sglang-url "$SGLANG_URL" \
    --temperature "${TEMPERATURE:-0.7}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-200}"
