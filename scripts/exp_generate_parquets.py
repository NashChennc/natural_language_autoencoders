#!/usr/bin/env python
"""Batch parquet generator for NLA prompt-comparison experiments.

Reads a YAML config describing multiple prompt variants and calls
make_qwen_layer20_demo_parquet.py as a subprocess for each one.
Output goes to $NLA_TMP_ROOT/exp/{plan}/{variant}.parquet.

Usage:
    conda activate nla
    source scripts/qwen_nla_env.sh
    python scripts/exp_generate_parquets.py --config docs/exp/plan_a.yaml
    python scripts/exp_generate_parquets.py --config docs/exp/plan_a.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def resolve_default(key: str, config_value, env_var: str, fallback):
    """Return config_value if not None, else env_var if set, else fallback."""
    if config_value is not None:
        return config_value
    env_val = os.environ.get(env_var)
    if env_val is not None:
        return env_val
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Path to experiment YAML config (e.g. docs/exp/plan_a.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print subprocess commands without executing them")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text())

    plan_name = cfg["plan"]
    desc = cfg.get("description", "(no description)")
    print(f"Plan: {plan_name}")
    print(f"Description: {desc}")
    print()

    nla_tmp_root = os.environ.get("NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp")
    output_root = cfg.get("output_root", f"{nla_tmp_root}/exp")
    output_dir = Path(output_root) / plan_name
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = cfg.get("defaults", {})

    base_model = resolve_default(
        "base_model", defaults.get("base_model"),
        "QWEN_BASE_MODEL",
        "/NAS/chennc/shared/models/Qwen/Qwen2.5-7B-Instruct",
    )
    layer_index = defaults.get("layer_index", 20)
    reply_tokens = defaults.get("reply_tokens", 80)
    max_rows = defaults.get("max_rows", 16)

    script_dir = Path(__file__).resolve().parent
    generator_script = script_dir / "make_qwen_layer20_demo_parquet.py"
    if not generator_script.exists():
        print(f"ERROR: generator script not found: {generator_script}",
              file=sys.stderr)
        sys.exit(1)

    n_total = len(cfg["variants"])
    failures = 0
    for idx, variant in enumerate(cfg["variants"], 1):
        name = variant["name"]
        prompt = variant.get("prompt", defaults.get("prompt", ""))
        system = variant.get("system", defaults.get("system"))
        if system is None:
            system = ""

        out_path = output_dir / f"{name}.parquet"

        cmd = [
            sys.executable, str(generator_script),
            "--base-model", str(base_model),
            "--output", str(out_path),
            "--prompt", prompt,
            "--system", system,
            "--layer-index", str(layer_index),
            "--reply-tokens", str(reply_tokens),
            "--max-rows", str(max_rows),
        ]

        print(f"[{idx}/{n_total}] {plan_name}/{name}")
        print(f"  system: {system[:80]}{'...' if len(system) > 80 else ''}")
        print(f"  prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  output: {out_path}")

        if args.dry_run:
            print(f"  [DRY RUN] would run: {' '.join(cmd)}")
            print()
            continue

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: subprocess failed with code {e.returncode}")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
            failures += 1
        print()

    if failures:
        print(f"Done with {failures} failure(s). {n_total - failures}/{n_total} parquets written to {output_dir}")
        sys.exit(1)
    print(f"Done. {n_total} parquets written to {output_dir}")


if __name__ == "__main__":
    main()
