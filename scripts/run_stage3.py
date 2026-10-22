"""Stage 3 runner: chains steps in one resumable process.

Examples
  GPU chain : python scripts/run_stage3.py --config configs/full.yaml --steps sample,shortcut,wait:corrupt,inferA,tables
  CPU job   : python scripts/run_stage3.py --config configs/full.yaml --steps sample,corrupt
`wait:<marker>` blocks until outputs/stage3/markers/<marker>.done exists (written by another process), so the GPU
never waits for a human between jobs. Every step skips work that already has a done-marker.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import pandas as pd
    from sfd.config import load_config, resolve, set_seed
    from sfd import stage3 as s3

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg, "out_dir")
    p = s3.P(cfg, out_dir)
    df = pd.read_parquet(p["stage1"])
    smp = None
    for step in a.steps.split(","):
        t0 = time.time()
        print(f"[run_stage3] {step} started", flush=True)
        if step == "sample":
            smp = s3.make_sample(cfg, p["out"], df)
        elif step == "shortcut":
            s3.run_shortcut(cfg, p, df, smp)
        elif step == "shortcut_v2":
            s3.run_shortcut_v2(cfg, p, df, smp)
        elif step == "corrupt":
            s3.run_corruptions(cfg, p, df, smp)
        elif step == "inferA":
            s3.run_infer_a(cfg, p, smp)
        elif step == "tables":
            s3.run_tables(cfg, p, df, smp)
        elif step.startswith("wait:"):
            mk = s3.marker(p["out"], step.split(":", 1)[1])
            while not mk.exists():
                time.sleep(60)
        else:
            raise SystemExit(f"unknown step {step}")
        print(f"[run_stage3] {step} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

