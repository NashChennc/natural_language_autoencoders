#!/usr/bin/env python
"""Prepare a manual claim-review CSV from original-case scored decodes."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _claim_type(text: str) -> str:
    low = text.lower()
    if any(w in low for w in ("intend", "plan", "prefer", "want", "refuse", "policy", "safety")):
        return "cognitive"
    if any(w in low for w in ("format", "bullet", "list", "question", "answer", "response")):
        return "format"
    if any(ch.isdigit() for ch in text) or any(w in low for w in ("qwen", "python", "javascript", "chinese", "english")):
        return "entity"
    return "context"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--min-cos", type=float, default=0.0)
    args = ap.parse_args()

    scores_path = Path(args.scores)
    out_path = Path(args.output or scores_path.with_name("claim_review.csv"))

    rows: list[dict[str, str]] = []
    with scores_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for score_row in reader:
            try:
                cos = float(score_row.get("critic_cos", "nan"))
            except ValueError:
                cos = float("nan")
            if cos < args.min_cos:
                continue
            explanation = score_row.get("explanation", "").strip()
            parts = [p.strip(" -\t") for p in SENTENCE_RE.split(explanation) if p.strip(" -\t")]
            for claim_idx, claim in enumerate(parts):
                if len(claim) < 8:
                    continue
                rows.append(
                    {
                        "claim_id": f"{score_row.get('experiment')}_{score_row.get('condition')}_{score_row.get('row_index')}_{claim_idx}",
                        "experiment": score_row.get("experiment", ""),
                        "condition": score_row.get("condition", ""),
                        "sample_strategy": score_row.get("sample_strategy", ""),
                        "row_index": score_row.get("row_index", ""),
                        "critic_cos": score_row.get("critic_cos", ""),
                        "claim_text": claim,
                        "claim_type_guess": _claim_type(claim),
                        "truth_label": "",
                        "specificity": "",
                        "recurring": "",
                        "review_notes": "",
                    }
                )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "claim_id", "experiment", "condition", "sample_strategy", "row_index",
            "critic_cos", "claim_text", "claim_type_guess", "truth_label",
            "specificity", "recurring", "review_notes",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} claims)")


if __name__ == "__main__":
    main()
