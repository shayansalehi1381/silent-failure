"""Stage 5 runner: python scripts/run_stage5.py --config configs/full.yaml  (implements docs/analysis_plan_frozen.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from sfd import stage5
    from sfd.config import load_config, resolve, set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    stage5.run(cfg, resolve(cfg, "out_dir"))
