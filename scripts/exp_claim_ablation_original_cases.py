#!/usr/bin/env python
"""Run AR-score deletion ablations for manually reviewed claims.

Input is `claim_review.csv` plus `scores.csv`. Rows with truth_label in
{true,false} are ablated by deleting `claim_text` from the original
explanation, then rescoring with the AR critic.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from nla_inference import NLACritic


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["experiment"], row["condition"], row["row_index"])


def _load_vector(data_root: Path, experiment: str, condition: str, row_index: int) -> np.ndarray:
    parquet_path = data_root / experiment / f"{condition}.parquet"
    table = pq.read_table(str(parquet_path), columns=["activation_vector"])
    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(
        zero_copy_only=False,
    ).astype(np.float32)
    n = table.num_rows
    vecs = flat.reshape(n, flat.shape[0] // n)
    return vecs[row_index]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--critic", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    nla_tmp_root = os.environ.get("NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp")
    data_root = Path(args.data_root or f"{nla_tmp_root}/exp/original_cases_l20_2026_05_22")
    critic_path = args.critic or os.environ.get(
        "QWEN_NLA_AR",
        "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar",
    )
    out_path = Path(args.output or Path(args.claims).with_name("claim_ablation.csv"))

    scores: dict[tuple[str, str, str], dict[str, str]] = {}
    with Path(args.scores).open(newline="") as f:
        for row in csv.DictReader(f):
            scores[_row_key(row)] = row

    critic = NLACritic(critic_path, device=args.device)
    out_rows: list[dict[str, object]] = []
    with Path(args.claims).open(newline="") as f:
        for claim in csv.DictReader(f):
            label = claim.get("truth_label", "").strip().lower()
            if label not in {"true", "false"}:
                continue
            key = (claim["experiment"], claim["condition"], claim["row_index"])
            score_row = scores.get(key)
            if not score_row:
                continue
            explanation = score_row["explanation"]
            claim_text = claim["claim_text"]
            ablated = explanation.replace(claim_text, "").strip()
            if ablated == explanation:
                continue
            vec = _load_vector(data_root, claim["experiment"], claim["condition"], int(claim["row_index"]))
            base_mse = float(score_row["critic_mse"])
            base_cos = float(score_row["critic_cos"])
            ablated_mse, ablated_cos = critic.score(ablated, vec)
            out_rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "experiment": claim["experiment"],
                    "condition": claim["condition"],
                    "row_index": claim["row_index"],
                    "truth_label": label,
                    "claim_text": claim_text,
                    "base_mse": base_mse,
                    "base_cos": base_cos,
                    "ablated_mse": ablated_mse,
                    "ablated_cos": ablated_cos,
                    "delta_mse": ablated_mse - base_mse,
                    "delta_cos": ablated_cos - base_cos,
                }
            )

    with out_path.open("w", newline="") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else [
            "claim_id", "experiment", "condition", "row_index", "truth_label",
            "claim_text", "base_mse", "base_cos", "ablated_mse", "ablated_cos",
            "delta_mse", "delta_cos",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out_path} ({len(out_rows)} ablations)")


if __name__ == "__main__":
    main()
