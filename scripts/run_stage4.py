"""Stage 4 runner (monitoring references and signals): python scripts/run_stage4.py --config configs/full.yaml
--steps refs,pilot,signals,tables. Each step skips work that already has a done-marker; 'refs' writes the
refs_frozen marker before 'signals' may read any shifted-family file."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import pandas as pd
    from sfd import stage4 as s4
    from sfd.config import load_config, resolve, set_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="refs,pilot,signals,tables")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg, "out_dir")
    p = s4.P(cfg, out_dir)
    df = pd.read_parquet(p["stage1"])
    smp = pd.read_csv(out_dir / "stage3" / "sample_id_test_3000.csv")
    for step in a.steps.split(","):
        t0 = time.time()
        print(f"[run_stage4] {step} started", flush=True)
        {"refs": lambda: s4.fit_references(cfg, p, df), "pilot": lambda: s4.run_pilot(cfg, p, df, smp),
         "signals": lambda: s4.run_signals(cfg, p, df, smp), "tables": lambda: s4.run_tables(cfg, p, df, smp)}[step]()
        print(f"[run_stage4] {step} done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
