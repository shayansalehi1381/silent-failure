"""Phase 11a runner: python scripts/run_stage11.py --config configs/full.yaml (accepted-only sensitivity; the CAM case study is scripts/make_cam_cases.py)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from sfd import stage11
    from sfd.config import load_config, resolve, set_seed
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    stage11.run(cfg, resolve(cfg, "out_dir"))
