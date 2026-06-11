#!/usr/bin/env python
"""Generate position-resolved activation parquet for NLA.

Extracts layer-K activation vectors from EARLY, MID, and LATE windows within
a single base model generation, to study how the residual stream evolves.

Unlike the standard generator (which only saves positions 0..max_rows from the
reply start), this extracts from three windows spanning the full generation.

Output columns:
  n_raw_tokens, abs_position, window, token_text,
  detokenized_text_truncated, activation_vector, activation_layer,
  doc_id, source

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/make_position_resolved_parquet.py \
        --base-model "$QWEN_BASE_MODEL" \
        --output "$NLA_TMP_ROOT/exp/position_resolved/C1_benign_cats.parquet" \
        --prompt "Tell me about cats" --system "" \
        --layer-index 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WINDOW_SIZE = 16


def _schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("n_raw_tokens", pa.int64()),
            ("abs_position", pa.int64()),
            ("window", pa.string()),
            ("token_text", pa.string()),
            ("detokenized_text_truncated", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
            ("source", pa.string()),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--layer-index", type=int, default=20)
    ap.add_argument("--reply-tokens", type=int, default=180,
                    help="Max new tokens (longer replies = better mid/late windows)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to("cuda").eval()

    # Build prompt
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    prompt_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    prompt_len = prompt_ids.shape[1]

    # Generate reply (greedy)
    with torch.inference_mode():
        full_ids = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device),
            max_new_tokens=args.reply_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    full_len = full_ids.shape[1]
    reply_len = full_len - prompt_len
    reply_text = tok.decode(full_ids[0, prompt_len:], skip_special_tokens=True)

    # Gather full reply token texts for manifest
    reply_tokens_list = [
        tok.decode([int(full_ids[0, i])], skip_special_tokens=False)
        for i in range(prompt_len, full_len)
    ]

    # Hook target layer
    layers = model.model.layers
    assert 0 <= args.layer_index < len(layers)
    captured: torch.Tensor | None = None

    def hook(_module, _inputs, output) -> None:
        nonlocal captured
        o = output[0] if isinstance(output, tuple) else output
        captured = o.detach().float().cpu()

    handle = layers[args.layer_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            full_mask = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
            model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    finally:
        handle.remove()

    assert captured is not None, "hook did not fire"
    hidden = captured[0]  # [T, d_model]
    d_model = hidden.shape[-1]

    # Define position windows
    # Early: 0..15  (same as Round 1)
    # Mid:   reply_len//2 .. reply_len//2 + 15
    # Late:  reply_len - 16 .. reply_len - 1  (or as close as possible)
    early_start = prompt_len
    mid_start = prompt_len + max(0, reply_len // 2 - WINDOW_SIZE // 2)
    late_start = max(prompt_len, full_len - WINDOW_SIZE)
    # Clamp to valid range
    late_start = max(prompt_len, min(late_start, full_len - 1))

    windows = [
        ("early", list(range(early_start, min(early_start + WINDOW_SIZE, full_len)))),
        ("mid", list(range(mid_start, min(mid_start + WINDOW_SIZE, full_len)))),
        ("late", list(range(late_start, min(late_start + WINDOW_SIZE, full_len)))),
    ]

    rows = {
        "n_raw_tokens": [],
        "abs_position": [],
        "window": [],
        "token_text": [],
        "detokenized_text_truncated": [],
        "activation_vector": [],
        "activation_layer": [],
        "doc_id": [],
        "source": [],
    }

    for window_name, positions in windows:
        for pos in positions:
            if pos >= full_len:
                continue
            rows["n_raw_tokens"].append(pos + 1)
            rows["abs_position"].append(pos)
            rows["window"].append(window_name)
            rows["token_text"].append(
                tok.decode([int(full_ids[0, pos])], skip_special_tokens=False)
            )
            rows["detokenized_text_truncated"].append(
                tok.decode(full_ids[0, :pos + 1], skip_special_tokens=True)
            )
            rows["activation_vector"].append(hidden[pos].tolist())
            rows["activation_layer"].append(args.layer_index)
            rows["doc_id"].append("position_resolved")
            rows["source"].append("base_model_greedy_reply_windowed")

    pq.write_table(pa.Table.from_pydict(rows, schema=_schema(d_model)), out_path)

    # Detailed manifest
    manifest = {
        "base_model": args.base_model,
        "layer_index": args.layer_index,
        "d_model": d_model,
        "prompt": args.prompt,
        "system": args.system,
        "prompt_tokens": prompt_len,
        "full_tokens": full_len,
        "reply_tokens": reply_len,
        "rows": len(rows["window"]),
        "windows": {
            "early": f"positions {windows[0][1][0]}-{windows[0][1][-1] if windows[0][1] else 'none'} "
                     f"({len(windows[0][1])} vectors)",
            "mid": f"positions {windows[1][1][0]}-{windows[1][1][-1] if windows[1][1] else 'none'} "
                   f"({len(windows[1][1])} vectors)",
            "late": f"positions {windows[2][1][0]}-{windows[2][1][-1] if windows[2][1] else 'none'} "
                    f"({len(windows[2][1])} vectors)",
        },
        "reply_text": reply_text,
        "reply_tokens_list": reply_tokens_list,
        "parquet": str(out_path),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Reply length: {reply_len} tokens")
    for wname, wpos in windows:
        print(f"  {wname}: positions {wpos[0] if wpos else 'none'}-{wpos[-1] if wpos else 'none'} "
              f"({len(wpos)} vectors)")
    print(f"Wrote {len(rows['window'])} rows -> {out_path}")
    print(f"Manifest -> {manifest_path}")
    print(f"Base reply:\n{reply_text[:300]}...")


if __name__ == "__main__":
    main()
