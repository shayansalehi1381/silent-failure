"""Export small reliability-bin tables (id_test, 10 equal-width confidence bins, per seed, before/after temperature scaling) from the
saved Stage 2 predictions, so the final notebook can redraw reliability diagrams with bin counts without per-row arrays.
python scripts/export_reliability_bins.py --config configs/full.yaml  ->  outputs/stage2/reliability_bins.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sfd import calibration as cal
    from sfd.config import load_config, resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = resolve(cfg, "out_dir") / "stage2"
    rows = []
    for ds in ("pathmnist", "camelyon17"):
        for seed in cfg["train"]["seeds"]:
            z = np.load(out / "preds" / f"{ds}_seed{seed}_id_test.npz")
            lo, y, T = z["logits"], z["labels"], float(z["T"])
            for variant, t in (("before temperature scaling", 1.0), ("after temperature scaling", T)):
                c, a, n = cal.reliability_curve(cal.probs(lo, t), y, n_bins=10)
                for b in range(10):
                    rows.append({"dataset": ds, "seed": seed, "variant": variant, "bin": b, "bin_conf": c[b], "bin_acc": a[b], "count": int(n[b])})
    pd.DataFrame(rows).round(6).to_csv(out / "reliability_bins.csv", index=False)
    print("wrote", out / "reliability_bins.csv", len(rows), "rows")
