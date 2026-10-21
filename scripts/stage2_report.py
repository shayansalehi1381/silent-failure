"""Build outputs/stage2/per_seed_results.md (+ .csv) from the saved stage-2 metric files (numbers are only read, never typed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfd.config import ROOT  # noqa: E402
from sfd.data.audit import to_markdown  # noqa: E402

out = ROOT / "outputs" / "stage2"
SHIFT = {"pathmnist": ["nct7k_test"], "camelyon17": ["ood_c1", "ood_c2"]}
blocks, flat = [], []
for ds in ("pathmnist", "camelyon17"):
    m = pd.read_csv(out / f"metrics_per_seed_{ds}.csv")
    c = pd.read_csv(out / f"calibration_{ds}.csv").set_index("seed")
    rows = []
    for seed in sorted(m["seed"].unique()):
        r = {"seed": seed, "best_epoch": int(c.loc[seed, "best_epoch"]), "T": f"{c.loc[seed, 'T_standard']:.3f}"}
        for split in ("val_cal", "id_test"):
            u = m[(m.seed == seed) & (m.split == split) & (m.variant == "uncalibrated")].iloc[0]
            t = m[(m.seed == seed) & (m.split == split) & (m.variant == "T_standard")].iloc[0]
            r[f"{split} acc"] = f"{t.acc:.4f}"
            r[f"{split} NLL"] = f"{u.nll:.4f} -> {t.nll:.4f}"
            r[f"{split} ECE"] = f"{u.ece:.4f} -> {t.ece:.4f}"
            r[f"{split} Brier"] = f"{u.brier:.4f} -> {t.brier:.4f}"
        r["id_test AUROC"] = f"{t.auroc:.4f}"
        for split in SHIFT[ds]:
            u = m[(m.seed == seed) & (m.split == split) & (m.variant == "uncalibrated")].iloc[0]
            t = m[(m.seed == seed) & (m.split == split) & (m.variant == "T_standard")].iloc[0]
            r[f"{split} acc"], r[f"{split} AUROC"] = f"{t.acc:.4f}", f"{t.auroc:.4f}"
            r[f"{split} NLL"], r[f"{split} ECE"] = f"{u.nll:.4f} -> {t.nll:.4f}", f"{u.ece:.4f} -> {t.ece:.4f}"
        rows.append(r)
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", ds)
    flat.append(df)
    blocks.append(f"## {ds}: per seed (T fitted on val_cal only; 'a -> b' = before -> after temperature scaling)\n\n"
                  + to_markdown(df.drop(columns=["dataset"])))
pd.concat(flat).to_csv(out / "per_seed_results.csv", index=False)
(out / "per_seed_results.md").write_text("# Stage 2 per-seed results\n\n" + "\n".join(blocks), encoding="utf-8")
print("\n".join(blocks))
