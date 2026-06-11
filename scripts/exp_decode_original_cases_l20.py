#!/usr/bin/env python
"""Decode all original-case parquets with the Qwen L20 AV SGLang server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
from nla_inference import NLAClient


def _load_vectors(parquet_path: Path) -> np.ndarray:
    pf = pq.ParquetFile(str(parquet_path))
    table = pf.read(columns=["activation_vector"])
    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(
        zero_copy_only=False,
    ).astype(np.float32)
    n = table.num_rows
    return flat.reshape(n, flat.shape[0] // n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--only-experiment", default=None)
    ap.add_argument("--only-condition", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    nla_tmp_root = os.environ.get("NLA_TMP_ROOT", "/NAS/chennc/NashChennc/.tmp")
    data_root = Path(args.data_root or f"{nla_tmp_root}/exp/original_cases_l20_2026_05_22")
    actor = args.actor or os.environ.get(
        "QWEN_NLA_AV",
        "/NAS/chennc/shared/models/kitft/nla-qwen2.5-7b-L20-av",
    )

    parquets = sorted(data_root.glob("*/*.parquet"))
    if args.only_experiment:
        parquets = [p for p in parquets if p.parent.name == args.only_experiment]
    if args.only_condition:
        parquets = [p for p in parquets if p.stem == args.only_condition]
    if not parquets:
        raise SystemExit(f"no parquets found under {data_root}")

    client = NLAClient(actor, sglang_url=args.sglang_url)
    failures = 0
    for idx, parquet_path in enumerate(parquets, 1):
        log_path = parquet_path.with_suffix(".decode.log")
        if log_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(parquets)}] skip existing {log_path}")
            continue

        print(f"[{idx}/{len(parquets)}] decode {parquet_path}")
        try:
            vecs = _load_vectors(parquet_path)
            with log_path.open("w") as f:
                f.write(f"[decode] parquet={parquet_path}\n")
                f.write(f"[decode] rows={len(vecs)} temperature={args.temperature} max_new_tokens={args.max_new_tokens}\n")
                for row_idx, v in enumerate(vecs):
                    out = client.generate(
                        v,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                    f.write(f"─── [{row_idx}]  ||v||={np.linalg.norm(v):.1f} ─────────────────────\n")
                    f.write(out.strip() + "\n\n")
        except Exception as exc:
            failures += 1
            print(f"ERROR: failed {parquet_path}: {exc}", file=sys.stderr)

    if failures:
        raise SystemExit(f"decode completed with {failures} failure(s)")
    print("decode completed")


if __name__ == "__main__":
    main()
