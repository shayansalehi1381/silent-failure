"""Stage 10 runner: python scripts/run_stage10.py --config configs/full.yaml  (docs/extension_plan_frozen.md; E2 is computed in the notebook)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from sfd import stage10
    from sfd.config import load_config, resolve, set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    stage10.run(cfg, resolve(cfg, "out_dir"))
