"""Pipeline driver: runs enabled stages in order, resumable via per-stage done-markers.

Phase 0 skeleton: stage 0 (environment record) is real; stages 1-9 are placeholders that
are registered here as they are implemented in later phases.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfd.config import ROOT, load_config, resolve, set_seed  # noqa: E402


def stage0(cfg: dict, out_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from env_record import collect

    rec = collect(cfg["seed"], resolve(cfg, "data_dir"))
    (out_dir / "env_record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"  GPU: {rec.get('gpu', 'none')}, cuda_available={rec['cuda_available']}")


def stage1(cfg: dict, out_dir: Path) -> None:
    from sfd.data import stage1 as s1

    s1.run(cfg, out_dir)


STAGES = {0: stage0, 1: stage1}  # later phases register stage2..stage9 here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="skip stages with a done-marker")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    out_dir = resolve(cfg, "out_dir")
    (out_dir / "markers").mkdir(parents=True, exist_ok=True)

    for s in cfg.get("stages", []):
        marker = out_dir / "markers" / f"stage{s}.done"
        if a.resume and marker.exists():
            print(f"[stage {s}] done earlier -> skipped (--resume)")
            continue
        if s not in STAGES:
            raise SystemExit(f"stage {s} is enabled in config but not implemented yet")
        print(f"[stage {s}] running")
        t0 = time.time()
        STAGES[s](cfg, out_dir)
        marker.write_text(f"{time.time() - t0:.1f}s\n")
        print(f"[stage {s}] ok ({time.time() - t0:.1f}s)")
    print("run_all finished")


if __name__ == "__main__":
    main()
