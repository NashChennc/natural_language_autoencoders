#!/usr/bin/env python
"""Quantitative evaluation for position-resolved (windowed) activation parquets.

Scores every vector in early/mid/late windows, grouping by condition and window.

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/exp_position_resolved_eval.py
"""

from __future__ import annotations

import argparse
import csv
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


def extract_explanations(decode_log_path: Path) -> list[str]:
    text = decode_log_path.read_text()
    blocks = re.split(r'^───\s*\[\d+\].*$', text, flags=re.MULTILINE)
    explanations = []
    for block in blocks:
        block = block.strip()
        if not block or block.startswith('[NLAClient]') or block.startswith('[smoke]'):
            continue
        if 'WARNING' in block:
            continue
        explanations.append(block)
    return explanations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--critic", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    critic_path = args.critic or os.environ.get(
        "QWEN_NLA_AR",
        "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar"
    )
    data_dir = Path(args.data_dir or os.environ.get(
        "NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp"
    )) / "exp" / "position_resolved"

    print(f"Loading Critic from {critic_path} on {args.device}...")
    critic = NLACritic(critic_path, device=args.device)
    print(f"Critic ready. mse_scale={critic.mse_scale:.2f}\n")

    rows = []
    condition_stats = {}

    for pq_path in sorted(data_dir.glob("*.parquet")):
        if pq_path.name.endswith('.manifest.json'):
            continue
        log_path = pq_path.with_suffix('.decode.log')
        if not log_path.exists():
            print(f"  SKIP {pq_path.name}: no decode log")
            continue

        condition = pq_path.stem
        print(f"Processing {condition}...")

        expls = extract_explanations(log_path)
        if not expls:
            print(f"  SKIP {condition}: no explanations extracted")
            continue

        pf = pq.ParquetFile(str(pq_path))
        batch = next(pf.iter_batches(batch_size=len(expls),
                     columns=["activation_vector", "window", "abs_position"]))
        n = min(len(expls), len(batch))

        flat = batch.column("activation_vector").flatten().to_numpy(
            zero_copy_only=False).astype(np.float32)
        d = flat.shape[0] // len(batch)
        vecs = flat.reshape(len(batch), d)

        windows = batch.column("window").to_pylist()[:n]
        positions = batch.column("abs_position").to_pylist()[:n]

        for i in range(n):
            v = vecs[i]
            expl = expls[i]
            w = windows[i]
            pos = positions[i]
            nm = float(np.linalg.norm(v))
            try:
                mse, cos = critic.score(expl, v)
            except Exception:
                mse, cos = float('nan'), float('nan')

            rows.append({
                "condition": condition, "window": w,
                "abs_position": pos, "norm": nm,
                "critic_mse": mse, "critic_cos": cos,
            })

        # Per-window stats
        for w in ["early", "mid", "late"]:
            w_rows = [r for r in rows
                      if r["condition"] == condition and r["window"] == w]
            if not w_rows:
                continue
            w_cos = [r["critic_cos"] for r in w_rows if not np.isnan(r["critic_cos"])]
            w_norm = [r["norm"] for r in w_rows]
            condition_stats[f"{condition}/{w}"] = {
                "n": len(w_rows),
                "cos_mean": statistics.mean(w_cos) if w_cos else float('nan'),
                "cos_std": statistics.stdev(w_cos) if len(w_cos) > 1 else 0,
                "norm_mean": statistics.mean(w_norm) if w_norm else float('nan'),
            }

    # Write CSV
    csv_path = data_dir / "position_scores.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "window", "abs_position", "norm",
            "critic_mse", "critic_cos",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {csv_path} ({len(rows)} rows)")

    # Write report
    md_path = data_dir / "position_report.md"
    lines = [
        "# Position-Resolved Quantitative Report",
        "",
        f"**Conditions:** 4 (C1 benign, C2 refusal, C3 fiction, C5 jailbreak)",
        f"**Windows:** early (0-15), mid (N/2), late (N-16..N-1)",
        f"**Total vectors scored:** {len(rows)}",
        "",
        "## Per-Condition × Window Statistics",
        "",
        "| Condition/Window | N | cos mean | cos σ | norm mean |",
        "|---|---|---|---|---|",
    ]
    for name, stats in sorted(condition_stats.items()):
        cos_str = f"{stats['cos_mean']:.3f}" if not np.isnan(stats['cos_mean']) else "N/A"
        lines.append(
            f"| {name} | {stats['n']} | {cos_str} | "
            f"{stats['cos_std']:.3f} | {stats['norm_mean']:.1f} |"
        )

    # Key comparison: C2 early vs mid vs late cos
    lines.append("")
    lines.append("## C2 (Refusal): Representational Crystallization")
    lines.append("")
    for w in ["early", "mid", "late"]:
        key = f"C2_dangerous_molotov/{w}"
        if key in condition_stats:
            s = condition_stats[key]
            lines.append(f"- **{w}**: cos={s['cos_mean']:.3f}, norm={s['norm_mean']:.1f}")

    lines.append("")
    lines.append("## C1 (Benign): Baseline Stability")
    lines.append("")
    for w in ["early", "mid", "late"]:
        key = f"C1_benign_cats/{w}"
        if key in condition_stats:
            s = condition_stats[key]
            lines.append(f"- **{w}**: cos={s['cos_mean']:.3f}, norm={s['norm_mean']:.1f}")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
