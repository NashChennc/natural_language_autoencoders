#!/usr/bin/env python
"""Gemma Scope-style reconstruction metrics for original-case L20 NLA runs.

This evaluates the existing original-case experiment outputs with three metrics:

* NLA L0 proxy: explanation token count under the AR tokenizer. This is not SAE
  latent L0; it is a transparent description-length proxy for NLA.
* FVU: scale-matched activation reconstruction SSE divided by a dataset-mean
  baseline SSE. The AR critic is direction-trained, so the primary NLA FVU first
  rescales each prediction to the corresponding original activation norm.
* delta LM loss: next-token cross-entropy increase after splicing the
  scale-matched reconstructed layer-20 vector back into the base LM forward pass
  at the sampled sequence position.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from nla_inference import NLACritic  # noqa: E402


DEFAULT_DATA_ROOT = "/NAS/chennc/NashChennc/.tmp/exp/original_cases_l20_2026_05_22"
DEFAULT_BASE_MODEL = "/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CRITIC = "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar"


@dataclass
class EvalRow:
    idx: int
    score_row: dict[str, str]
    parquet_path: Path
    row_index: int

    @property
    def experiment(self) -> str:
        return self.score_row["experiment"]

    @property
    def condition(self) -> str:
        return self.score_row["condition"]

    @property
    def strategy(self) -> str:
        return self.score_row["sample_strategy"]

    @property
    def explanation(self) -> str:
        return self.score_row.get("explanation", "")

    @property
    def abs_position(self) -> int:
        return int(self.score_row["abs_position"])

    @property
    def reply_position(self) -> int:
        return int(self.score_row["reply_position"])


def _parse_csv_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {part.strip() for part in value.split(",") if part.strip()}
    return out or None


def _default_stem(experiments: set[str] | None, max_rows_per_condition: int | None) -> str:
    stem = "gemmascope_reconstruction_metrics"
    if experiments:
        stem += "_" + "_".join(sorted(experiments))
    if max_rows_per_condition is not None:
        stem += f"_max{max_rows_per_condition}"
    return stem


def parquet_from_score_row(data_root: Path, row: dict[str, str]) -> Path:
    decode_log = row.get("decode_log", "")
    if decode_log:
        log_path = Path(decode_log)
        name = log_path.name
        if name.endswith(".decode.log"):
            return log_path.with_name(name[: -len(".decode.log")] + ".parquet")
    return data_root / row["experiment"] / f"{row['condition']}.parquet"


def load_eval_rows(
    scores_path: Path,
    data_root: Path,
    experiments: set[str] | None,
    max_rows_per_condition: int | None,
) -> list[EvalRow]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[EvalRow] = []
    with scores_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if experiments and raw["experiment"] not in experiments:
                continue
            key = (raw["experiment"], raw["condition"])
            if max_rows_per_condition is not None and counts[key] >= max_rows_per_condition:
                continue
            counts[key] += 1
            rows.append(
                EvalRow(
                    idx=len(rows),
                    score_row=raw,
                    parquet_path=parquet_from_score_row(data_root, raw),
                    row_index=int(raw["row_index"]),
                )
            )
    return rows


def load_parquet_vectors(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(str(parquet_path), columns=["activation_vector"])
    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(
        zero_copy_only=False,
    ).astype(np.float32)
    n = table.num_rows
    return flat.reshape(n, flat.shape[0] // n)


def load_original_vectors(rows: list[EvalRow]) -> np.ndarray:
    by_path: dict[Path, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_path[row.parquet_path].append(row)

    vectors: list[np.ndarray | None] = [None] * len(rows)
    for path, path_rows in sorted(by_path.items(), key=lambda kv: str(kv[0])):
        vecs = load_parquet_vectors(path)
        for row in path_rows:
            vectors[row.idx] = vecs[row.row_index]

    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        raise RuntimeError(f"missing original vectors for row indices: {missing[:10]}")
    return np.stack([v for v in vectors if v is not None]).astype(np.float32)


def reconstruct_or_load(
    rows: list[EvalRow],
    critic_path: Path,
    cache_path: Path,
    *,
    device: str,
    overwrite: bool,
    batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    if cache_path.exists() and not overwrite:
        recon = np.load(cache_path).astype(np.float32)
        if recon.shape[0] != len(rows):
            raise RuntimeError(
                f"cache row count mismatch: {cache_path} has {recon.shape[0]}, "
                f"current selection has {len(rows)}"
            )
        tokenizer = AutoTokenizer.from_pretrained(str(critic_path), trust_remote_code=True)
        l0_proxy = [
            len(tokenizer(row.explanation, add_special_tokens=False)["input_ids"])
            for row in rows
        ]
        return recon, l0_proxy

    critic = NLACritic(str(critic_path), device=device)
    if critic.tokenizer.pad_token_id is None:
        critic.tokenizer.pad_token = critic.tokenizer.eos_token
    d_model = critic.value_head.out_features
    recon = np.full((len(rows), d_model), np.nan, dtype=np.float32)
    l0_proxy = [
        len(critic.tokenizer(row.explanation, add_special_tokens=False)["input_ids"])
        for row in rows
    ]

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [critic.template.format(explanation=row.explanation) for row in batch_rows]
        try:
            encoded = critic.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(critic.device)
            attention_mask = encoded["attention_mask"].to(critic.device)
            with torch.inference_mode():
                hidden = critic.backbone.model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state
                last_positions = attention_mask.sum(dim=1) - 1
                selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
                pred = critic.value_head(selected).float().cpu().numpy().astype(np.float32)
            recon[start : start + len(batch_rows)] = pred
        except Exception as exc:
            print(f"[reconstruct] batch start={start} size={len(batch_rows)} failed: {exc}")
            for offset, row in enumerate(batch_rows):
                try:
                    recon[start + offset] = critic.reconstruct(row.explanation).numpy().astype(np.float32)
                except Exception as row_exc:
                    print(
                        f"[reconstruct] row={start + offset} "
                        f"{row.experiment}/{row.condition} failed: {row_exc}"
                    )
        done = min(start + len(batch_rows), len(rows))
        if done % 50 == 0 or done == len(rows):
            print(f"[reconstruct] {done}/{len(rows)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, recon)
    return recon, l0_proxy


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def scale_match_recon(originals: np.ndarray, recon: np.ndarray) -> np.ndarray:
    out = np.full_like(recon, np.nan, dtype=np.float32)
    pred_norm = np.linalg.norm(recon, axis=1)
    gold_norm = np.linalg.norm(originals, axis=1)
    ok = np.isfinite(recon).all(axis=1) & (pred_norm > 0.0) & (gold_norm > 0.0)
    out[ok] = recon[ok] / pred_norm[ok, None] * gold_norm[ok, None]
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.mean(vals) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(vals) if vals else float("nan")


def _format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_ids_from_manifest(tokenizer: AutoTokenizer, manifest_path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    messages = []
    if manifest.get("system"):
        messages.append({"role": "system", "content": manifest["system"]})
    messages.append({"role": "user", "content": manifest["prompt"]})
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].tolist()

    reply_ids: list[int] = []
    reencoded_from_token_list = True
    for token_text in manifest.get("reply_tokens_list", []):
        encoded = tokenizer.encode(token_text, add_special_tokens=False)
        if len(encoded) != 1:
            reencoded_from_token_list = False
            reply_ids = tokenizer.encode(manifest.get("reply_text", ""), add_special_tokens=False)
            break
        reply_ids.append(int(encoded[0]))

    if not reply_ids and manifest.get("reply_text"):
        reencoded_from_token_list = False
        reply_ids = tokenizer.encode(manifest["reply_text"], add_special_tokens=False)

    ids = prompt_ids + reply_ids
    diag = {
        "manifest_prompt_tokens": int(manifest.get("prompt_tokens", -1)),
        "manifest_full_tokens": int(manifest.get("full_tokens", -1)),
        "rebuilt_prompt_tokens": len(prompt_ids),
        "rebuilt_full_tokens": len(ids),
        "prompt_token_match": len(prompt_ids) == int(manifest.get("prompt_tokens", -1)),
        "full_token_match": len(ids) == int(manifest.get("full_tokens", -1)),
        "reply_from_token_list": reencoded_from_token_list,
    }
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0), diag


def _patch_layer_output(output, positions: torch.Tensor, vectors: torch.Tensor):
    hidden = output[0] if isinstance(output, tuple) else output
    patched = hidden.clone()
    batch_index = torch.arange(patched.shape[0], device=patched.device)
    patched[batch_index, positions] = vectors.to(device=patched.device, dtype=patched.dtype)
    if isinstance(output, tuple):
        return (patched,) + output[1:]
    return patched


@torch.inference_mode()
def compute_delta_lm_loss(
    rows: list[EvalRow],
    patch_vectors: np.ndarray,
    data_root: Path,
    base_model_path: Path,
    *,
    layer_index: int,
    device: str,
    batch_size: int,
) -> dict[int, dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    layers = model.model.layers
    if not 0 <= layer_index < len(layers):
        raise RuntimeError(f"layer_index={layer_index} out of range for {len(layers)} layers")

    by_condition: dict[tuple[str, str, Path], list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_condition[(row.experiment, row.condition, row.parquet_path)].append(row)

    out: dict[int, dict[str, object]] = {}
    for cond_i, ((exp, cond, parquet_path), cond_rows) in enumerate(
        sorted(by_condition.items(), key=lambda kv: (kv[0][0], kv[0][1])),
        start=1,
    ):
        manifest_path = parquet_path.with_suffix(parquet_path.suffix + ".manifest.json")
        input_ids_cpu, diag = build_ids_from_manifest(tokenizer, manifest_path)
        seq_len = int(input_ids_cpu.shape[1])
        input_ids = input_ids_cpu.to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        cond_rows_valid = [row for row in cond_rows if 0 <= row.abs_position < seq_len - 1]
        skipped = len(cond_rows) - len(cond_rows_valid)
        if skipped:
            print(f"[delta] {exp}/{cond}: skipped {skipped} rows beyond rebuilt sequence")

        base_loss_by_pos: dict[int, float] = {}
        for pos in sorted({row.abs_position for row in cond_rows_valid}):
            target = input_ids[0, pos + 1].unsqueeze(0)
            base_loss_by_pos[pos] = float(F.cross_entropy(logits[0, pos].float().unsqueeze(0), target))

        for start in range(0, len(cond_rows_valid), batch_size):
            batch_rows = cond_rows_valid[start : start + batch_size]
            bsz = len(batch_rows)
            batch_ids = input_ids.repeat(bsz, 1)
            batch_mask = attention_mask.repeat(bsz, 1)
            positions = torch.tensor([row.abs_position for row in batch_rows], dtype=torch.long, device=device)
            targets = batch_ids[torch.arange(bsz, device=device), positions + 1]
            patch_vecs = torch.tensor(
                np.stack([patch_vectors[row.idx] for row in batch_rows]).astype(np.float32),
                device=device,
            )

            handle = layers[layer_index].register_forward_hook(
                lambda _module, _inputs, output, positions=positions, patch_vecs=patch_vecs: (
                    _patch_layer_output(output, positions, patch_vecs)
                )
            )
            try:
                patched_logits = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    use_cache=False,
                ).logits
            finally:
                handle.remove()

            gathered = patched_logits[torch.arange(bsz, device=device), positions].float()
            recon_losses = F.cross_entropy(gathered, targets, reduction="none")
            for j, row in enumerate(batch_rows):
                base_loss = base_loss_by_pos[row.abs_position]
                recon_loss = float(recon_losses[j])
                out[row.idx] = {
                    "base_lm_loss": base_loss,
                    "recon_lm_loss": recon_loss,
                    "delta_lm_loss": recon_loss - base_loss,
                    **diag,
                }

        for row in cond_rows:
            if row.idx not in out:
                out[row.idx] = {
                    "base_lm_loss": float("nan"),
                    "recon_lm_loss": float("nan"),
                    "delta_lm_loss": float("nan"),
                    **diag,
                }

        print(
            f"[delta] {cond_i}/{len(by_condition)} {exp}/{cond}: "
            f"rows={len(cond_rows_valid)} seq_len={seq_len} "
            f"token_match={diag['full_token_match']}"
        )

    return out


def add_reconstruction_metrics(
    rows: list[EvalRow],
    originals: np.ndarray,
    recon: np.ndarray,
    scale_matched_recon: np.ndarray,
    l0_proxy: list[int],
    delta: dict[int, dict[str, object]] | None,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        gold = originals[row.idx]
        pred = recon[row.idx]
        pred_sm = scale_matched_recon[row.idx]
        finite = bool(np.isfinite(pred).all())
        if finite:
            diff = gold - pred
            sse = float(np.dot(diff, diff))
            mse = float(sse / diff.shape[0])
            cos = _cosine(gold, pred)
            pred_norm = float(np.linalg.norm(pred))
        else:
            sse = mse = cos = pred_norm = float("nan")

        finite_sm = bool(np.isfinite(pred_sm).all())
        if finite_sm:
            diff_sm = gold - pred_sm
            sm_sse = float(np.dot(diff_sm, diff_sm))
            sm_mse = float(sm_sse / diff_sm.shape[0])
            sm_cos = _cosine(gold, pred_sm)
            sm_norm = float(np.linalg.norm(pred_sm))
        else:
            sm_sse = sm_mse = sm_cos = sm_norm = float("nan")

        drow = delta.get(row.idx, {}) if delta is not None else {}
        base = float(drow.get("base_lm_loss", float("nan")))
        recon_loss = float(drow.get("recon_lm_loss", float("nan")))
        delta_loss = float(drow.get("delta_lm_loss", float("nan")))

        result = dict(row.score_row)
        result.update(
            {
                "metric_row_index": row.idx,
                "nla_l0_proxy_tokens": int(l0_proxy[row.idx]),
                "raw_recon_sse": sse,
                "raw_recon_mse": mse,
                "raw_recon_cos": cos,
                "raw_recon_norm": pred_norm,
                "scale_matched_recon_sse": sm_sse,
                "scale_matched_recon_mse": sm_mse,
                "scale_matched_recon_cos": sm_cos,
                "scale_matched_recon_norm": sm_norm,
                "base_lm_loss": base,
                "recon_lm_loss": recon_loss,
                "delta_lm_loss": delta_loss,
            }
        )
        for key, value in drow.items():
            if key not in result:
                result[key] = value
        out.append(result)
    return out


def group_fvu(
    metric_rows: list[dict[str, object]],
    originals: np.ndarray,
    *,
    sse_key: str,
) -> dict[tuple[str, ...], float]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(metric_rows):
        if not math.isfinite(float(row[sse_key])):
            continue
        groups[("overall",)].append(i)
        groups[("experiment", str(row["experiment"]))].append(i)
        groups[("condition", str(row["experiment"]), str(row["condition"]))].append(i)
        groups[("strategy", str(row["sample_strategy"]))].append(i)

    fvus: dict[tuple[str, ...], float] = {}
    for key, idxs in groups.items():
        if len(idxs) < 2:
            fvus[key] = float("nan")
            continue
        xs = originals[idxs].astype(np.float64)
        den = float(((xs - xs.mean(axis=0, keepdims=True)) ** 2).sum())
        num = sum(float(metric_rows[i][sse_key]) for i in idxs)
        fvus[key] = num / den if den > 0.0 else float("nan")
    return fvus


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    label: tuple[str, ...],
    rows: list[dict[str, object]],
    fvu: float,
) -> str:
    delta_vals = [float(r["delta_lm_loss"]) for r in rows]
    return (
        f"| {' / '.join(label)} | {len(rows)} | "
        f"{_mean(float(r['nla_l0_proxy_tokens']) for r in rows):.1f} | "
        f"{_format_float(fvu, 4)} | "
        f"{_format_float(_mean(delta_vals), 4)} | "
        f"{_format_float(_median(delta_vals), 4)} | "
        f"{_format_float(_mean(float(r['scale_matched_recon_cos']) for r in rows), 4)} | "
        f"{_format_float(_mean(float(r['scale_matched_recon_mse']) for r in rows), 2)} |"
    )


def write_summary(
    metric_rows: list[dict[str, object]],
    originals: np.ndarray,
    path: Path,
    *,
    base_model: Path,
    critic: Path,
    layer_index: int,
    skipped_delta: bool,
) -> None:
    fvus = group_fvu(metric_rows, originals, sse_key="scale_matched_recon_sse")
    raw_fvus = group_fvu(metric_rows, originals, sse_key="raw_recon_sse")

    by_experiment: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    by_condition: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    by_strategy: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_experiment[(str(row["experiment"]),)].append(row)
        by_condition[(str(row["experiment"]), str(row["condition"]))].append(row)
        by_strategy[(str(row["sample_strategy"]),)].append(row)

    lines = [
        "# Gemma Scope-style Reconstruction Metrics for Original Cases L20",
        "",
        f"- Base model: `{base_model}`",
        f"- AR critic/reconstructor: `{critic}`",
        f"- Layer patched for delta LM loss: `{layer_index}`",
        f"- Rows: `{len(metric_rows)}`",
        "",
        "## Metric Mapping",
        "",
        "| Metric | SAE/Gemma Scope meaning | NLA measurement here | Comparability caveat |",
        "|---|---|---|---|",
        "| mean L0 | Mean number of active SAE latents per token | `nla_l0_proxy_tokens`: explanation token count under AR tokenizer | Not a sparse latent count; use only as NLA description-length/cost proxy |",
        "| delta LM loss | CE increase after replacing activations with SAE reconstructions in the LM forward pass | Next-token CE increase after replacing the sampled layer-20 vector with the scale-matched AR reconstruction | Position-sampled local proxy, not full-sequence all-token SAE replacement |",
        "| FVU | Reconstruction SSE normalized by dataset-mean SSE | Scale-matched activation SSE normalized by the same selected rows' mean-activation baseline | AR is direction-trained; raw prediction norm is diagnostic, not the primary NLA fidelity score |",
        "",
        "Primary NLA FVU and delta LM loss use AR predictions rescaled to the original activation norm. Raw reconstruction columns remain in the CSV as diagnostics.",
        "",
    ]

    if skipped_delta:
        lines += [
            "> Delta LM loss was skipped in this run; the delta columns are `NA`.",
            "",
        ]

    overall_rows = metric_rows
    lines += [
        "## Overall",
        "",
        "| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _summary_row(("overall",), overall_rows, fvus.get(("overall",), float("nan"))),
        "",
        f"- Raw-norm diagnostic overall FVU: `{_format_float(raw_fvus.get(('overall',), float('nan')), 4)}`",
        "",
        "## By Experiment",
        "",
        "| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, rows in sorted(by_experiment.items()):
        lines.append(_summary_row(key, rows, fvus.get(("experiment", *key), float("nan"))))

    lines += [
        "",
        "## By Condition",
        "",
        "| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, rows in sorted(by_condition.items()):
        lines.append(_summary_row(key, rows, fvus.get(("condition", *key), float("nan"))))

    lines += [
        "",
        "## By Sampling Strategy",
        "",
        "| Group | N | mean L0 proxy | FVU | mean delta LM loss | median delta LM loss | mean scale-matched cos | mean scale-matched MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, rows in sorted(by_strategy.items()):
        lines.append(_summary_row(key, rows, fvus.get(("strategy", *key), float("nan"))))

    token_match_rows = [
        row for row in metric_rows if "full_token_match" in row and str(row["full_token_match"]) == "False"
    ]
    if token_match_rows:
        lines += [
            "",
            "## Tokenization Warnings",
            "",
            f"- Rows with rebuilt full-token length mismatch: `{len(token_match_rows)}`",
        ]

    path.write_text("\n".join(lines) + "\n")


def read_metric_csvs(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    rows.sort(
        key=lambda r: (
            str(r["experiment"]),
            str(r["condition"]),
            int(r["row_index"]),
            str(r["sample_strategy"]),
            str(r.get("event_keyword", "")),
        )
    )
    return rows


def eval_rows_from_metric_rows(data_root: Path, metric_rows: list[dict[str, object]]) -> list[EvalRow]:
    out: list[EvalRow] = []
    for i, row in enumerate(metric_rows):
        score_row = {str(k): str(v) for k, v in row.items()}
        out.append(
            EvalRow(
                idx=i,
                score_row=score_row,
                parquet_path=parquet_from_score_row(data_root, score_row),
                row_index=int(score_row["row_index"]),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=os.environ.get("NLA_ORIGINAL_CASES_ROOT", DEFAULT_DATA_ROOT))
    ap.add_argument("--scores", default=None)
    ap.add_argument("--base-model", default=os.environ.get("QWEN_BASE_MODEL", DEFAULT_BASE_MODEL))
    ap.add_argument("--critic", default=os.environ.get("QWEN_NLA_AR", DEFAULT_CRITIC))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer-index", type=int, default=20)
    ap.add_argument("--experiments", default=None, help="Comma-separated subset, e.g. A,B,C")
    ap.add_argument("--max-rows-per-condition", type=int, default=None)
    ap.add_argument("--batch-size-reconstruct", type=int, default=16)
    ap.add_argument("--batch-size-delta", type=int, default=4)
    ap.add_argument("--skip-delta-lm-loss", action="store_true")
    ap.add_argument("--overwrite-recon-cache", action="store_true")
    ap.add_argument("--output-stem", default=None)
    ap.add_argument(
        "--merge-inputs",
        default=None,
        help="Comma-separated metric CSVs to merge and summarize without rerunning models.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    scores_path = Path(args.scores) if args.scores else data_root / "scores.csv"
    base_model = Path(args.base_model)
    critic = Path(args.critic)
    experiments = _parse_csv_list(args.experiments)
    stem = args.output_stem or _default_stem(experiments, args.max_rows_per_condition)

    if args.merge_inputs:
        input_paths = [Path(p.strip()) for p in args.merge_inputs.split(",") if p.strip()]
        metric_rows = read_metric_csvs(input_paths)
        eval_rows = eval_rows_from_metric_rows(data_root, metric_rows)
        originals = load_original_vectors(eval_rows)
        out_csv = data_root / f"{stem}.csv"
        out_md = data_root / f"{stem}_summary.md"
        write_csv(metric_rows, out_csv)
        write_summary(
            metric_rows,
            originals,
            out_md,
            base_model=base_model,
            critic=critic,
            layer_index=args.layer_index,
            skipped_delta=False,
        )
        print(f"merged rows: {len(metric_rows)}")
        print(f"wrote {out_csv}")
        print(f"wrote {out_md}")
        return

    rows = load_eval_rows(scores_path, data_root, experiments, args.max_rows_per_condition)
    if not rows:
        raise SystemExit("no rows selected")
    print(f"selected rows: {len(rows)}")

    originals = load_original_vectors(rows)
    recon_cache = data_root / f"{stem}_recon_vectors.npy"
    recon, l0_proxy = reconstruct_or_load(
        rows,
        critic,
        recon_cache,
        device=args.device,
        overwrite=args.overwrite_recon_cache,
        batch_size=args.batch_size_reconstruct,
    )
    scale_matched_recon = scale_match_recon(originals, recon)

    delta = None
    if not args.skip_delta_lm_loss:
        delta = compute_delta_lm_loss(
            rows,
            scale_matched_recon,
            data_root,
            base_model,
            layer_index=args.layer_index,
            device=args.device,
            batch_size=args.batch_size_delta,
        )

    metric_rows = add_reconstruction_metrics(
        rows,
        originals,
        recon,
        scale_matched_recon,
        l0_proxy,
        delta,
    )
    out_csv = data_root / f"{stem}.csv"
    out_md = data_root / f"{stem}_summary.md"
    write_csv(metric_rows, out_csv)
    write_summary(
        metric_rows,
        originals,
        out_md,
        base_model=base_model,
        critic=critic,
        layer_index=args.layer_index,
        skipped_delta=args.skip_delta_lm_loss,
    )
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"wrote {recon_cache}")


if __name__ == "__main__":
    main()
