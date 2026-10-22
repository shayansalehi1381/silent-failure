"""Stage 8 runner: python scripts/run_stage8.py --config configs/full.yaml [--steps augment,main,labels,tables] (docs/ablation_plan_frozen.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from sfd import stage8
    from sfd.config import load_config, resolve, set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="augment,main,labels,tables")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    stage8.run(cfg, resolve(cfg, "out_dir"), a.steps)
