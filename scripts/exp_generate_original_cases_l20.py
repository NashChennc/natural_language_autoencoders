#!/usr/bin/env python
"""Batch-generate parquets for the layer-20 original-case suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _resolve(value, env_name: str, fallback: str) -> str:
    if value:
        return str(value)
    return os.environ.get(env_name, fallback)


def _join(values: list[str] | None) -> str:
    return "||".join(values or [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="docs/exp/original_cases_l20.yaml")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--only-experiment", default=None,
                    help="Run one experiment id, e.g. A")
    ap.add_argument("--only-condition", default=None,
                    help="Run one condition id, e.g. A1_real_user")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    defaults = cfg.get("defaults", {})

    nla_tmp_root = os.environ.get("NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp")
    output_root = Path(args.output_root or f"{nla_tmp_root}/exp/original_cases_l20_2026_05_22")
    base_model = _resolve(
        defaults.get("base_model"),
        "QWEN_BASE_MODEL",
        "/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct",
    )
    system_default = defaults.get("system", "")
    layer_index = int(defaults.get("layer_index", 20))
    if layer_index != 20:
        raise SystemExit(f"ERROR: this suite is layer-20-only, got layer_index={layer_index}")

    script = Path(__file__).resolve().parent / "make_original_case_l20_parquet.py"
    suite = cfg["suite"]
    total = 0
    failures = 0

    for exp in cfg["experiments"]:
        exp_id = exp["id"]
        if args.only_experiment and args.only_experiment != exp_id:
            continue
        for cond in exp["conditions"]:
            cond_id = cond["id"]
            if args.only_condition and args.only_condition != cond_id:
                continue

            total += 1
            out_path = output_root / exp_id / f"{cond_id}.parquet"
            cmd = [
                sys.executable, str(script),
                "--base-model", base_model,
                "--output", str(out_path),
                "--suite", suite,
                "--experiment", exp_id,
                "--condition", cond_id,
                "--matched-control", str(cond.get("matched_control", "")),
                "--system", str(cond.get("system", system_default) or ""),
                "--prompt", str(cond["prompt"]),
                "--layer-index", str(layer_index),
                "--reply-tokens", str(cond.get("reply_tokens", defaults.get("reply_tokens", 180))),
                "--sample-strategies", _join(cond.get("sample_strategies", defaults.get("sample_strategies", []))),
                "--event-keywords", _join(cond.get("event_keywords", [])),
            ]
            if args.dry_run:
                cmd.append("--dry-run")

            print(f"[{total}] {exp_id}/{cond_id} -> {out_path}")
            try:
                result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                print(result.stdout)
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(f"ERROR: {exp_id}/{cond_id} failed with code {exc.returncode}", file=sys.stderr)
                print(exc.stdout, file=sys.stderr)
                print(exc.stderr, file=sys.stderr)

    if failures:
        raise SystemExit(f"Done with {failures} failure(s) across {total} condition(s).")
    print(f"Done. {total} condition(s) processed under {output_root}")


if __name__ == "__main__":
    main()
