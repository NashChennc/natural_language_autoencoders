#!/usr/bin/env python
"""Build a small real-activation parquet for Qwen NLA inference.

The output contains layer-20 residual-stream vectors from tokens in Qwen's
own greedy reply. It is intentionally small so it can be regenerated on one
A40 before starting the AV SGLang server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
DEFAULT_USER = "What are you hiding?"


def _schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("n_raw_tokens", pa.int64()),
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
    ap.add_argument("--prompt", default=DEFAULT_USER)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--layer-index", type=int, default=20)
    ap.add_argument("--reply-tokens", type=int, default=80)
    ap.add_argument("--max-rows", type=int, default=16)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    prompt_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device)

    with torch.inference_mode():
        full_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=args.reply_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    reply_text = tok.decode(full_ids[0, prompt_len:], skip_special_tokens=True)

    layers = model.model.layers
    assert 0 <= args.layer_index < len(layers), (
        f"layer_index={args.layer_index} out of range for {len(layers)} layers"
    )
    captured: torch.Tensor | None = None

    def hook(_module, _inputs, output) -> None:
        nonlocal captured
        captured = output[0] if isinstance(output, tuple) else output
        captured = captured.detach().float().cpu()

    handle = layers[args.layer_index].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            full_mask = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
            model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    finally:
        handle.remove()

    assert captured is not None, f"forward hook on layer {args.layer_index} did not fire"
    hidden = captured[0]
    d_model = hidden.shape[-1]

    reply_positions = list(range(prompt_len, full_len))[: args.max_rows]
    rows = {
        "n_raw_tokens": [],
        "token_text": [],
        "detokenized_text_truncated": [],
        "activation_vector": [],
        "activation_layer": [],
        "doc_id": [],
        "source": [],
    }
    for pos in reply_positions:
        rows["n_raw_tokens"].append(pos + 1)
        rows["token_text"].append(tok.decode([int(full_ids[0, pos])], skip_special_tokens=False))
        rows["detokenized_text_truncated"].append(
            tok.decode(full_ids[0, : pos + 1], skip_special_tokens=True)
        )
        rows["activation_vector"].append(hidden[pos].tolist())
        rows["activation_layer"].append(args.layer_index)
        rows["doc_id"].append("qwen_demo_reply")
        rows["source"].append("base_model_greedy_reply")

    pq.write_table(pa.Table.from_pydict(rows, schema=_schema(d_model)), out_path)

    manifest = {
        "base_model": args.base_model,
        "layer_index": args.layer_index,
        "d_model": d_model,
        "prompt": args.prompt,
        "system": args.system,
        "prompt_tokens": prompt_len,
        "full_tokens": full_len,
        "rows": len(reply_positions),
        "reply_text": reply_text,
        "parquet": str(out_path),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {len(reply_positions)} rows -> {out_path}")
    print(f"manifest -> {manifest_path}")
    print("base reply:")
    print(reply_text)


if __name__ == "__main__":
    main()
