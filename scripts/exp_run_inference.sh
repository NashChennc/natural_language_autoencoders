#!/usr/bin/env bash
# exp_run_inference.sh — Run NLA decode on all parquets in an experiment plan
# directory. Assumes SGLang server is already running at $SGLANG_URL.
#
# Usage:
#     conda activate nla
#     source scripts/qwen_nla_env.sh
#     bash scripts/launch_qwen_nla_sglang.sh &   # start SGLang first
#     bash scripts/exp_run_inference.sh plan_a
#     bash scripts/exp_run_inference.sh plan_a --temperature 0.0 --n-rows 8
#
# Environment variables (all optional):
#     SGLANG_URL         default http://localhost:30000
#     N_ROWS             default 16
#     TEMPERATURE        default 0.0
#     MAX_NEW_TOKENS     default 200

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/qwen_nla_env.sh"

# ─── Parse arguments ──────────────────────────────────────────────────────────
PLAN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --temperature)
            TEMPERATURE="$2"; shift 2 ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"; shift 2 ;;
        --n-rows)
            N_ROWS="$2"; shift 2 ;;
        --sglang-url)
            SGLANG_URL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 <plan_name> [--temperature T] [--max-new-tokens N] [--n-rows N] [--sglang-url URL]"
            exit 0 ;;
        -*)
            echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        *)
            PLAN="$1"; shift ;;
    esac
done

if [[ -z "$PLAN" ]]; then
    echo "ERROR: plan name required. Usage: $0 <plan_name>" >&2
    exit 1
fi

# ─── Defaults ─────────────────────────────────────────────────────────────────
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
N_ROWS="${N_ROWS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"

# ─── Locate parquets ─────────────────────────────────────────────────────────
EXP_DIR="$NLA_TMP_ROOT/exp/$PLAN"
if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: experiment directory not found: $EXP_DIR" >&2
    echo "Run 'python scripts/exp_generate_parquets.py --config docs/exp/${PLAN}.yaml' first." >&2
    exit 1
fi

PARQUETS=()
while IFS= read -r -d '' f; do
    PARQUETS+=("$f")
done < <(find "$EXP_DIR" -maxdepth 1 -name '*.parquet' -not -name '*.manifest.json' -print0 | sort -z)

if [[ ${#PARQUETS[@]} -eq 0 ]]; then
    echo "ERROR: no .parquet files found in $EXP_DIR" >&2
    exit 1
fi

echo "=== NLA Batch Inference ==="
echo "Plan:        $PLAN"
echo "Exp dir:     $EXP_DIR"
echo "SGLang URL:  $SGLANG_URL"
echo "Parquets:    ${#PARQUETS[@]}"
echo "Temperature: $TEMPERATURE"
echo "N rows:      $N_ROWS"
echo "Max tokens:  $MAX_NEW_TOKENS"
echo "AV ckpt:     $QWEN_NLA_AV"
echo "==========================="
echo

# ─── Run inference on each parquet ────────────────────────────────────────────
N_TOTAL=${#PARQUETS[@]}
failures=0
for i in "${!PARQUETS[@]}"; do
    parquet="${PARQUETS[$i]}"
    base="$(basename "$parquet" .parquet)"
    log="$EXP_DIR/${base}.decode.log"

    idx=$((i + 1))
    echo -n "[$idx/$N_TOTAL] $base ... "

    if python "$REPO_ROOT/nla_inference.py" "$QWEN_NLA_AV" \
        --sglang-url "$SGLANG_URL" \
        --parquet "$parquet" \
        --n "$N_ROWS" \
        --temperature "$TEMPERATURE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        > "$log" 2>&1; then

        if grep -q '^───' "$log" 2>/dev/null; then
            n_lines=$(grep -c '^───' "$log" || true)
            echo "OK ($n_lines decodes)"
        else
            echo "WARNING: no decode output lines in log"
        fi
    else
        echo "FAILED (see $log)"
        failures=$((failures + 1))
    fi
done

echo
if [[ $failures -gt 0 ]]; then
    echo "Done with $failures failure(s). $((N_TOTAL - failures))/$N_TOTAL succeeded."
else
    echo "Done. $N_TOTAL parquets decoded successfully."
fi
echo "Logs: $EXP_DIR/*.decode.log"
