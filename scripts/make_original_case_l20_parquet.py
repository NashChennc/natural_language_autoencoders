#!/usr/bin/env python
"""Generate layer-20 activation parquet for original-case NLA experiments.

This generator keeps layer fixed at 20 by default and supports the sampling
strategies needed for the original case/limitations suite:

  prompt_final, reply_early, reply_mid, reply_late, event_window, uniform_50

It writes one parquet plus a manifest for a single condition. Batch orchestration
is handled by scripts/exp_generate_original_cases_l20.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WINDOW_SIZE = 16


@dataclass(frozen=True)
class SampledPosition:
    abs_position: int
    reply_position: int
    sample_strategy: str
    window: str
    event_keyword: str


def _schema(d_model: int) -> pa.Schema:
    return pa.schema(
        [
            ("suite", pa.string()),
            ("experiment", pa.string()),
            ("condition", pa.string()),
            ("matched_control", pa.string()),
            ("sample_strategy", pa.string()),
            ("window", pa.string()),
            ("event_keyword", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("abs_position", pa.int64()),
            ("reply_position", pa.int64()),
            ("token_text", pa.string()),
            ("detokenized_text_truncated", pa.string()),
            ("activation_vector", pa.list_(pa.float32(), d_model)),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.string()),
            ("source", pa.string()),
        ]
    )


def _window_positions(start: int, stop: int, *, strategy: str, window: str) -> list[SampledPosition]:
    return [
        SampledPosition(pos, -1, strategy, window, "")
        for pos in range(max(start, 0), max(stop, start))
    ]


def _reply_window(
    prompt_len: int,
    full_len: int,
    reply_len: int,
    strategy: str,
) -> list[SampledPosition]:
    if reply_len <= 0:
        return []
    if strategy == "reply_early":
        start = prompt_len
    elif strategy == "reply_mid":
        start = prompt_len + max(0, reply_len // 2 - WINDOW_SIZE // 2)
    elif strategy == "reply_late":
        start = max(prompt_len, full_len - WINDOW_SIZE)
    else:
        raise ValueError(f"not a reply window strategy: {strategy}")

    stop = min(start + WINDOW_SIZE, full_len)
    return [
        SampledPosition(pos, pos - prompt_len, strategy, strategy.removeprefix("reply_"), "")
        for pos in range(start, stop)
    ]


def _uniform_positions(prompt_len: int, full_len: int, reply_len: int, n: int = 50) -> list[SampledPosition]:
    if reply_len <= 0:
        return []
    if reply_len <= n:
        reply_positions = list(range(reply_len))
    else:
        # Deterministic integer grid including the first and last reply tokens.
        reply_positions = sorted({round(i * (reply_len - 1) / (n - 1)) for i in range(n)})
    return [
        SampledPosition(prompt_len + rp, rp, "uniform_50", "uniform", "")
        for rp in reply_positions
        if prompt_len + rp < full_len
    ]


def _event_positions(
    reply_tokens: list[str],
    prompt_len: int,
    full_len: int,
    keywords: list[str],
) -> list[SampledPosition]:
    if not keywords:
        return []

    reply_text = "".join(reply_tokens)
    lower_text = reply_text.lower()
    out: list[SampledPosition] = []
    seen_events: set[tuple[str, int]] = set()

    # Map character offsets to approximate token positions by cumulative decoded text.
    cumulative: list[int] = []
    total = 0
    for tok in reply_tokens:
        total += len(tok)
        cumulative.append(total)

    for keyword in keywords:
        needle = keyword.lower()
        if not needle:
            continue
        char_idx = lower_text.find(needle)
        if char_idx < 0:
            continue
        token_idx = 0
        while token_idx < len(cumulative) and cumulative[token_idx] <= char_idx:
            token_idx += 1
        key = (keyword, token_idx)
        if key in seen_events:
            continue
        seen_events.add(key)

        start = max(0, token_idx - WINDOW_SIZE // 2)
        stop = min(len(reply_tokens), start + WINDOW_SIZE)
        for rp in range(start, stop):
            out.append(
                SampledPosition(
                    prompt_len + rp,
                    rp,
                    "event_window",
                    "event",
                    keyword,
                )
            )

    return [p for p in out if p.abs_position < full_len]


def _dedupe_positions(positions: list[SampledPosition]) -> list[SampledPosition]:
    # Keep duplicate absolute positions across strategies: sampling strategy is
    # the experimental variable for Experiment F. Only remove exact duplicates.
    seen: set[tuple[int, str, str, str]] = set()
    out: list[SampledPosition] = []
    for p in positions:
        key = (p.abs_position, p.sample_strategy, p.window, p.event_keyword)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _parse_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split("||") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="")
    ap.add_argument("--suite", default="original_cases_l20")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--matched-control", default="")
    ap.add_argument("--layer-index", type=int, default=20)
    ap.add_argument("--reply-tokens", type=int, default=180)
    ap.add_argument("--sample-strategies", default="prompt_final||reply_early||reply_mid||reply_late||event_window||uniform_50")
    ap.add_argument("--event-keywords", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    strategies = _parse_list(args.sample_strategies)
    event_keywords = _parse_list(args.event_keywords)
    out_path = Path(args.output)

    print(f"condition={args.experiment}/{args.condition}")
    print(f"output={out_path}")
    print(f"layer_index={args.layer_index}")
    print(f"strategies={strategies}")
    print(f"event_keywords={event_keywords}")
    print(f"prompt={args.prompt[:180]}{'...' if len(args.prompt) > 180 else ''}")
    if args.dry_run:
        return

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
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    prompt_ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    prompt_len = prompt_ids.shape[1]

    with torch.inference_mode():
        full_ids = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids, dtype=torch.long, device=prompt_ids.device),
            max_new_tokens=args.reply_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    full_len = full_ids.shape[1]
    reply_len = full_len - prompt_len
    reply_text = tok.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
    reply_tokens = [
        tok.decode([int(full_ids[0, pos])], skip_special_tokens=False)
        for pos in range(prompt_len, full_len)
    ]

    layers = model.model.layers
    assert 0 <= args.layer_index < len(layers), (
        f"layer_index={args.layer_index} out of range for {len(layers)} layers"
    )
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

    assert captured is not None, "forward hook did not fire"
    hidden = captured[0]
    d_model = hidden.shape[-1]

    positions: list[SampledPosition] = []
    if "prompt_final" in strategies:
        positions.append(SampledPosition(prompt_len - 1, -1, "prompt_final", "prompt_final", ""))
    for strategy in ("reply_early", "reply_mid", "reply_late"):
        if strategy in strategies:
            positions.extend(_reply_window(prompt_len, full_len, reply_len, strategy))
    if "event_window" in strategies:
        positions.extend(_event_positions(reply_tokens, prompt_len, full_len, event_keywords))
    if "uniform_50" in strategies:
        positions.extend(_uniform_positions(prompt_len, full_len, reply_len))

    positions = _dedupe_positions(positions)
    positions.sort(key=lambda p: (p.sample_strategy, p.abs_position, p.event_keyword))

    rows = {
        "suite": [],
        "experiment": [],
        "condition": [],
        "matched_control": [],
        "sample_strategy": [],
        "window": [],
        "event_keyword": [],
        "n_raw_tokens": [],
        "abs_position": [],
        "reply_position": [],
        "token_text": [],
        "detokenized_text_truncated": [],
        "activation_vector": [],
        "activation_layer": [],
        "doc_id": [],
        "source": [],
    }

    for p in positions:
        rows["suite"].append(args.suite)
        rows["experiment"].append(args.experiment)
        rows["condition"].append(args.condition)
        rows["matched_control"].append(args.matched_control)
        rows["sample_strategy"].append(p.sample_strategy)
        rows["window"].append(p.window)
        rows["event_keyword"].append(p.event_keyword)
        rows["n_raw_tokens"].append(p.abs_position + 1)
        rows["abs_position"].append(p.abs_position)
        rows["reply_position"].append(p.reply_position)
        rows["token_text"].append(tok.decode([int(full_ids[0, p.abs_position])], skip_special_tokens=False))
        rows["detokenized_text_truncated"].append(
            tok.decode(full_ids[0, : p.abs_position + 1], skip_special_tokens=True)
        )
        rows["activation_vector"].append(hidden[p.abs_position].tolist())
        rows["activation_layer"].append(args.layer_index)
        rows["doc_id"].append(f"{args.experiment}/{args.condition}")
        rows["source"].append("base_model_greedy_original_case_l20")

    pq.write_table(pa.Table.from_pydict(rows, schema=_schema(d_model)), out_path)

    manifest = {
        "suite": args.suite,
        "experiment": args.experiment,
        "condition": args.condition,
        "matched_control": args.matched_control,
        "base_model": args.base_model,
        "layer_index": args.layer_index,
        "d_model": d_model,
        "system": args.system,
        "prompt": args.prompt,
        "prompt_tokens": prompt_len,
        "full_tokens": full_len,
        "reply_tokens": reply_len,
        "rows": len(positions),
        "sample_strategies": strategies,
        "event_keywords": event_keywords,
        "reply_text": reply_text,
        "reply_tokens_list": reply_tokens,
        "parquet": str(out_path),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"reply_len={reply_len}")
    print(f"rows={len(positions)}")
    print(f"wrote={out_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
