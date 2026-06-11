#!/usr/bin/env python3
"""Extract representative NLA decode positions from scores.csv for LaTeX case files.

Outputs candidate position/explanation tables for each target condition.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

SCORES_CSV = Path(__file__).resolve().parent.parent / "docs/exp/completed/round3_original_cases/scores.csv"

# Conditions to extract, grouped by case file
TARGET_CONDITIONS = {
    "A": ["A1_real_user", "A2_benchmark_format", "A3_safety_eval_obvious",
          "A4_safety_eval_realistic", "A5_realism_edit"],
    "B": ["B1_neutral_baseline", "B3_hidden_policy", "B5_hidden_refusal"],
    "C": ["C3_switch_after_marker", "D3_force_wrong_style"],
    "D": ["D1_clean_math", "D2_wrong_label_hint", "D5_self_correction",
          "E2_high_specificity_pressure"],
    "E": ["E2_high_specificity_pressure", "E3_identity_template_control"],
    "F": ["F2_language_switch_sampling", "F3_answer_sampling"],
    "G": ["A2_benchmark_format"],
    "I": ["E4_identity_pressure"],
}

# Preferred strategies in order
PREFERRED_STRATEGIES = ["event_window", "reply_early", "reply_mid", "uniform_50", "reply_late", "prompt_final"]


def latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "°": r"\degree{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Replace double quotes with LaTeX quotes
    text = text.replace('"', "''")
    text = text.replace('"', "``")
    return text


def score_explanation(explanation: str) -> float:
    """Score an explanation's informativeness. Higher = more informative."""
    score = 0.0
    # Longer explanations tend to be more specific
    score += min(len(explanation) / 500, 1.0) * 2.0

    # Penalize generic "Structured ... format" starts
    generic_starts = [
        "Structured article format",
        "Structured response format",
        "Structured math format",
        "Structured format with",
        "Structured informational format",
        "Structured product description",
        "Structured technical answer",
        "Structured problem-solving",
        "Structured Python code",
        "Structured corporate policy",
        "Structured guidance format",
        "Structured platform UI",
        "Structured dictionary format",
    ]
    for gs in generic_starts:
        if explanation.startswith(gs):
            score -= 1.5
            break

    # Reward explanations that mention specific content (quoted text)
    if '("' in explanation or '("' in explanation or '``' in explanation:
        score += 1.0

    # Reward explanations that mention specific topics
    specific_keywords = [
        "refusal", "cannot", "I cannot", "safety", "evaluation", "benchmark",
        "language", "Chinese", "English", "SWITCH", "translation",
        "calculation", "arithmetic", "multiply", "PEMDAS",
        "confabulation", "false", "claim", "identity",
        "Python", "Java", "code", "debugging",
        "poem", "verse", "poet",
        "jailbreak", "DAN", "Molotov",
        "chemical", "explosive", "household",
        "tea", "culture", "Hangzhou",
        "cat", "dog", "pet",
        "hidden", "preference", "policy", "refuse",
    ]
    explanation_lower = explanation.lower()
    for kw in specific_keywords:
        if kw.lower() in explanation_lower:
            score += 0.5

    return score


def pick_best_positions(rows: list[dict], n: int = 3) -> list[dict]:
    """Pick the n best positions from rows, ensuring diversity."""
    # Score each row
    scored = [(score_explanation(r["explanation"]), r) for r in rows]
    scored.sort(key=lambda x: (-x[0], -float(x[1].get("critic_cos", 0))))

    # Pick diverse positions (different reply_position and strategy)
    picked = []
    seen_keys = set()
    for _, row in scored:
        key = (row["sample_strategy"], row["reply_position"])
        if key not in seen_keys or len(picked) < n // 2:
            picked.append(row)
            seen_keys.add(key)
        if len(picked) >= n:
            break

    # Sort by strategy priority then reply_position
    strat_order = {s: i for i, s in enumerate(PREFERRED_STRATEGIES)}
    picked.sort(key=lambda r: (
        strat_order.get(r["sample_strategy"], 99),
        int(r["reply_position"]) if r["reply_position"].lstrip('-').isdigit() else 999,
    ))

    return picked


def generate_latex_table(rows: list[dict], condition: str) -> str:
    """Generate LaTeX table rows for a condition."""
    lines = []
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lll}")
    lines.append(r"\toprule")
    lines.append(r"位置 & NLA decode & 备注 \\")
    lines.append(r"\midrule")

    for row in rows:
        strategy = row["sample_strategy"]
        rpos = row["reply_position"]
        explanation = row["explanation"]
        cos_val = float(row.get("critic_cos", 0))

        # Format position label
        if strategy == "prompt_final":
            pos_label = "prompt\\_final"
        else:
            pos_label = f"{strategy} pos {rpos}"

        # Truncate explanation to a reasonable length for the table
        if len(explanation) > 180:
            # Find a good break point
            break_at = explanation.rfind(". ", 80, 180)
            if break_at < 0:
                break_at = explanation.rfind(" ", 80, 180)
            if break_at > 80:
                explanation = explanation[:break_at] + "."

        # Generate a brief note
        note = f"cos={cos_val:.3f}" if cos_val > 0 else ""

        # Escape LaTeX
        explanation_escaped = latex_escape(explanation)
        note_escaped = latex_escape(note)

        lines.append(f"{pos_label} & {explanation_escaped} & {note_escaped} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found", file=sys.stderr)
        sys.exit(1)

    # Load all rows grouped by condition
    all_rows: dict[str, list[dict]] = defaultdict(list)
    with open(SCORES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]
            if condition in all_rows:
                pass
            all_rows[condition].append(row)

    # Process each target condition
    for case, conditions in TARGET_CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Case {case}")
        print(f"{'='*60}")

        for condition in conditions:
            rows = all_rows.get(condition, [])
            if not rows:
                print(f"\n  [{condition}] NO DATA")
                continue

            print(f"\n  --- {condition} ({len(rows)} rows) ---")

            # Pick best positions
            picked = pick_best_positions(rows, n=3)

            # Show what we picked
            for i, row in enumerate(picked):
                explanation = row["explanation"][:120]
                print(f"  [{i+1}] strategy={row['sample_strategy']} "
                      f"pos={row['reply_position']} "
                      f"cos={float(row.get('critic_cos',0)):.3f} "
                      f"norm={float(row.get('norm',0)):.1f}")
                print(f"      {explanation}...")

            # Generate LaTeX
            print(f"\n  [LATEX TABLE]")
            print(generate_latex_table(picked, condition))
            print()


if __name__ == "__main__":
    main()
