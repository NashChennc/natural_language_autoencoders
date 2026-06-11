#!/usr/bin/env python
"""Quantitative evaluation of NLA decodes using the AR Critic.

For every decode log + parquet pair across all experiment plans, computes:
  - Critic direction-MSE and cosine similarity (fidelity score)
  - Activation vector L2 norm
  - Per-condition aggregate statistics

Outputs:
  {exp_dir}/quant_report.md  — markdown summary
  {exp_dir}/quant_scores.csv — per-row data

Usage:
    conda activate nla
    source scripts/qwen_nla_env.sh
    CUDA_VISIBLE_DEVICES=7 python scripts/exp_quantitative_eval.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from pathlib import Path

# Allow importing nla_inference from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from nla_inference import NLACritic


def extract_explanations(decode_log_path: Path) -> list[str]:
    """Parse decode log, returning one explanation string per vector.

    The log format is:
      ─── [N]  ||v||=XXX ─────────────────────
      <explanation text, possibly multi-paragraph>

    Returns list of explanation strings, one per decoded vector.
    """
    text = decode_log_path.read_text()
    # Split on the separator pattern
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


def load_vectors(parquet_path: Path, n: int) -> np.ndarray:
    """Load activation vectors from parquet."""
    pf = pq.ParquetFile(str(parquet_path))
    batch = next(pf.iter_batches(batch_size=n, columns=["activation_vector"]))
    flat = batch.column("activation_vector").flatten().to_numpy(
        zero_copy_only=False).astype(np.float32)
    d = flat.shape[0] // len(batch)
    return flat.reshape(len(batch), d)


def compute_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda",
                    help="Device for Critic model (default: cuda — respects CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--critic", default=None,
                    help="Path to AR critic checkpoint (default: $QWEN_NLA_AR)")
    ap.add_argument("--exp-dir", default=None,
                    help="Experiment root (default: $NLA_TMP_ROOT/exp)")
    args = ap.parse_args()

    critic_path = args.critic or os.environ.get(
        "QWEN_NLA_AR",
        "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-ar"
    )
    exp_dir = Path(args.exp_dir or os.environ.get(
        "NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp"
    )) / "exp"

    print(f"Loading Critic from {critic_path} on {args.device}...")
    critic = NLACritic(critic_path, device=args.device)
    print(f"Critic ready. mse_scale={critic.mse_scale:.2f}\n")

    # Collect all (plan, variant, parquet, log) tuples
    all_plans = {}
    for plan_dir in sorted(exp_dir.glob("plan_*")):
        plan = plan_dir.name
        parquets = sorted(plan_dir.glob("*.parquet"))
        parquets = [p for p in parquets if not p.name.endswith('.manifest.json')]
        if not parquets:
            continue
        all_plans[plan] = []
        for pq_path in parquets:
            log_path = pq_path.with_suffix('.decode.log')
            manifest_path = Path(str(pq_path) + '.manifest.json')
            if not log_path.exists():
                continue
            all_plans[plan].append((pq_path, log_path, manifest_path))

    # Per-condition results
    rows = []
    plan_stats = {}

    for plan, items in all_plans.items():
        plan_mse = []
        plan_cos = []
        plan_norm = []
        for pq_path, log_path, manifest_path in items:
            variant = pq_path.stem
            expls = extract_explanations(log_path)
            if not expls:
                continue

            vecs = load_vectors(pq_path, len(expls))
            n = min(len(expls), len(vecs))

            var_mse = []
            var_cos = []
            var_norm = []
            for i in range(n):
                v = vecs[i]
                expl = expls[i]
                nm = compute_norm(v)
                try:
                    mse, cos = critic.score(expl, v)
                except Exception:
                    mse, cos = float('nan'), float('nan')

                rows.append({
                    "plan": plan, "variant": variant,
                    "position": i, "norm": nm,
                    "critic_mse": mse, "critic_cos": cos,
                })
                var_mse.append(mse)
                var_cos.append(cos)
                var_norm.append(nm)
                plan_mse.append(mse)
                plan_cos.append(cos)
                plan_norm.append(nm)

            valid_mse = [x for x in var_mse if not np.isnan(x)]
            valid_cos = [x for x in var_cos if not np.isnan(x)]
            plan_stats[f"{plan}/{variant}"] = {
                "n_vectors": n,
                "mse_mean": statistics.mean(valid_mse) if valid_mse else float('nan'),
                "mse_std": statistics.stdev(valid_mse) if len(valid_mse) > 1 else 0,
                "cos_mean": statistics.mean(valid_cos) if valid_cos else float('nan'),
                "cos_std": statistics.stdev(valid_cos) if len(valid_cos) > 1 else 0,
                "norm_mean": statistics.mean(var_norm) if var_norm else float('nan'),
                "norm_min": min(var_norm) if var_norm else float('nan'),
                "norm_max": max(var_norm) if var_norm else float('nan'),
            }

    # Write CSV
    csv_path = exp_dir / "quant_scores.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "plan", "variant", "position", "norm",
            "critic_mse", "critic_cos",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path} ({len(rows)} rows)")

    # Write Markdown report
    md_path = exp_dir / "quant_report.md"
    lines = [
        "# NLA Quantitative Evaluation Report",
        "",
        f"**Critic:** `{critic_path}`  \n"
        f"**MSE scale:** {critic.mse_scale:.2f} (√d_model, direction-only)  \n"
        f"**Total vectors scored:** {len(rows)}  \n"
        f"**Conditions:** {len(plan_stats)}",
        "",
        "## Per-Condition Statistics",
        "",
        "| Plan / Variant | N | cos mean | cos σ | MSE mean | MSE σ | norm mean | norm [min, max] |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name, stats in sorted(plan_stats.items()):
        cos_str = f"{stats['cos_mean']:.3f}" if not np.isnan(stats['cos_mean']) else "N/A"
        lines.append(
            f"| {name} | {stats['n_vectors']} | {cos_str} | "
            f"{stats['cos_std']:.3f} | {stats['mse_mean']:.3f} | "
            f"{stats['mse_std']:.3f} | {stats['norm_mean']:.1f} | "
            f"[{stats['norm_min']:.1f}, {stats['norm_max']:.1f}] |"
        )

    # Plan-level aggregates
    lines.append("")
    lines.append("## Plan-Level Aggregates")
    lines.append("")
    lines.append("| Plan | cos mean | MSE mean | norm mean |")
    lines.append("|---|---|---|---|")
    for plan in sorted(all_plans.keys()):
        plan_rows = [r for r in rows if r["plan"] == plan]
        plan_cos_vals = [r["critic_cos"] for r in plan_rows if not np.isnan(r["critic_cos"])]
        plan_mse_vals = [r["critic_mse"] for r in plan_rows if not np.isnan(r["critic_mse"])]
        plan_norm_vals = [r["norm"] for r in plan_rows]
        lines.append(
            f"| {plan} | {statistics.mean(plan_cos_vals):.3f} | "
            f"{statistics.mean(plan_mse_vals):.3f} | "
            f"{statistics.mean(plan_norm_vals):.1f} |"
        )

    # Plan C safety gradient
    lines.append("")
    lines.append("## Plan C: Safety Gradient (Key Test)")
    lines.append("")
    lines.append("| Variant | cos mean | norm mean | Interpretation |")
    lines.append("|---|---|---|---|")
    c_variants = ["C1_benign_cats", "C2_dangerous_molotov", "C3_creative_handcuffs",
                  "C4_harmful_speech", "C5_jailbreak"]
    for v in c_variants:
        name = f"plan_c/{v}"
        if name in plan_stats:
            s = plan_stats[name]
            lines.append(
                f"| {v} | {s['cos_mean']:.3f} | {s['norm_mean']:.1f} | "
                f"{'benign' if 'benign' in v else 'refusal' if 'molotov' in v else 'fiction' if 'handcuffs' in v else 'hedged' if 'harmful' in v else 'denial'} |"
            )

    lines.append("")
    lines.append("### Key Metric: Refusal vs. Benign Norm Ratio")
    if "plan_c/C1_benign_cats" in plan_stats and "plan_c/C2_dangerous_molotov" in plan_stats:
        c1_norm = plan_stats["plan_c/C1_benign_cats"]["norm_mean"]
        c2_norm = plan_stats["plan_c/C2_dangerous_molotov"]["norm_mean"]
        ratio = c2_norm / c1_norm
        lines.append(f"- C2 (refusal) / C1 (benign) norm ratio = **{ratio:.3f}**")
        lines.append(f"- If < 1.0, the norm depression effect is confirmed quantitatively.")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Report written: {md_path}")


if __name__ == "__main__":
    main()
