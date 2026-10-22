"""Stage 6 runner: python scripts/run_stage6.py --config configs/full.yaml --steps extract,windows,analysis."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import pandas as pd
    from sfd import stage6 as s6
    from sfd.config import load_config, resolve, set_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="extract,windows,analysis")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg, "out_dir")
    p = s6.P(cfg, out_dir)
    df = pd.read_parquet(p["stage1"])
    smp = pd.read_csv(out_dir / "stage3" / "sample_id_test_3000.csv")
    for step in a.steps.split(","):
        t0 = time.time()
        print(f"[run_stage6] {step} started", flush=True)
        {"extract": lambda: s6.extract(cfg, p, df, smp), "windows": lambda: s6.run_windows(cfg, p, df, smp),
         "analysis": lambda: s6.run_analysis(cfg, p, df, smp),
         "windows_a1": lambda: s6.run_windows(cfg, p, df, smp, variant="a1"),
         "analysis_a1": lambda: s6.run_analysis(cfg, p, df, smp, variant="a1")}[step]()
        print(f"[run_stage6] {step} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

