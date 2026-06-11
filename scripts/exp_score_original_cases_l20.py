#!/usr/bin/env python
"""Score original-case L20 decodes with the AR critic."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from nla_inference import NLACritic


SEP_RE = re.compile(r"^───\s*\[(\d+)\].*$", re.MULTILINE)


def extract_explanations(log_path: Path) -> list[str]:
    text = log_path.read_text()
    matches = list(SEP_RE.finditer(text))
    out: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        out.append(block)
    return out


def load_table(parquet_path: Path) -> tuple[list[dict[str, object]], np.ndarray]:
    table = pq.read_table(str(parquet_path))
    rows = table.drop(["activation_vector"]).to_pylist()
    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(
        zero_copy_only=False,
    ).astype(np.float32)
    n = table.num_rows
    vecs = flat.reshape(n, flat.shape[0] // n)
    return rows, vecs


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.mean(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--critic", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    nla_tmp_root = os.environ.get("NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp")
    data_root = Path(args.data_root or f"{nla_tmp_root}/exp/original_cases_l20_2026_05_22")
    critic_path = args.critic or os.environ.get(
        "QWEN_NLA_AR",
        "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar",
    )

    critic = NLACritic(critic_path, device=args.device)
    score_rows: list[dict[str, object]] = []

    for parquet_path in sorted(data_root.glob("*/*.parquet")):
        log_path = parquet_path.with_suffix(".decode.log")
        if not log_path.exists():
            print(f"skip missing decode log: {log_path}")
            continue

        meta_rows, vecs = load_table(parquet_path)
        explanations = extract_explanations(log_path)
        n = min(len(meta_rows), len(vecs), len(explanations))
        for i in range(n):
            expl = explanations[i]
            v = vecs[i]
            try:
                mse, cos = critic.score(expl, v)
            except Exception:
                mse, cos = float("nan"), float("nan")
            row = dict(meta_rows[i])
            row.update(
                {
                    "row_index": i,
                    "decode_log": str(log_path),
                    "explanation": expl,
                    "norm": float(np.linalg.norm(v)),
                    "critic_mse": mse,
                    "critic_cos": cos,
                }
            )
            score_rows.append(row)

    out_csv = data_root / "scores.csv"
    out_md = data_root / "scores_by_condition.md"
    if score_rows:
        fieldnames = list(score_rows[0].keys())
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(score_rows)
    else:
        out_csv.write_text("")

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in score_rows:
        key = (str(row["experiment"]), str(row["condition"]), str(row["sample_strategy"]))
        groups.setdefault(key, []).append(row)

    lines = [
        "# Original Cases L20 Scores",
        "",
        f"Critic: `{critic_path}`",
        f"Rows scored: {len(score_rows)}",
        "",
        "| Experiment | Condition | Strategy | N | cos mean | MSE mean | norm mean |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (exp, cond, strategy), rows in sorted(groups.items()):
        cos = _mean([float(r["critic_cos"]) for r in rows])
        mse = _mean([float(r["critic_mse"]) for r in rows])
        norm = _mean([float(r["norm"]) for r in rows])
        lines.append(f"| {exp} | {cond} | {strategy} | {len(rows)} | {cos:.3f} | {mse:.3f} | {norm:.1f} |")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
