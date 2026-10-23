"""Phase 11a (post-hoc / descriptive): qualitative Grad-CAM case study for the reversed-cue shortcut setting.

Selection rule (deterministic): among the rows of `stage11.cam_cases.model` in condition `reversed`, silent failures = wrong prediction with
calibrated confidence >= min_conf, correct = correct prediction; `numpy.random.default_rng(seed)` draws n_silent silent failures and n_correct
correct cases without replacement (uniformly, in table order). CAMs are the saved layer4 8x8 maps (outputs/stage6/rows, git-ignored); images
are re-read from the local cache and the reversed cue is pasted exactly as in the pipeline. No model is run.
python scripts/make_cam_cases.py --config configs/full.yaml -> outputs/stage11/cam_cases.csv + outputs/stage11/figures/fig7b_cam_cases.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle", "normal_mucosa", "stroma", "tumor_epithelium"]

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.ndimage import zoom

    from sfd import shortcut as sc
    from sfd import stage4 as s4
    from sfd.config import load_config, resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    c = cfg["stage11"]["cam_cases"]
    out_dir = resolve(cfg, "out_dir")
    p = s4.P(cfg, out_dir)
    p["s6"] = out_dir / "stage6"
    spec = next(s for s in s4.model_specs(cfg, p) if s["name"] == c["model"])
    z = np.load(p["s6"] / "rows" / f"{spec['name']}.npz")
    rows = np.flatnonzero((z["fam"] == "C") & (z["cond"] == c["condition"]))
    conf, pred, lab = z["conf"][rows], z["pred"][rows].astype(int), z["label"][rows].astype(int)
    silent = np.flatnonzero((pred != lab) & (conf >= c["min_conf"]))
    correct = np.flatnonzero(pred == lab)
    rng = np.random.default_rng(c["seed"])
    pick = list(rng.choice(silent, c["n_silent"], replace=False)) + list(rng.choice(correct, c["n_correct"], replace=False))
    kinds = ["silent failure"] * c["n_silent"] + ["correct"] * c["n_correct"]

    smp = pd.read_csv(out_dir / "stage3" / "sample_id_test_3000.csv")
    df = pd.read_parquet(p["stage1"])
    sets = s4._sets(cfg, p, df, spec, smp, None)
    st = next(s for s in sets if s["family"] == "C" and s["condition"] == c["condition"])
    assert np.array_equal(st["uid"], z["uid"][rows]), "set order differs from the saved rows"
    imgs = st["images"][st["pos"][pick]]
    imgs = sc.apply_cue_uint8(imgs, st["cue"][pick], sc.palette_for(spec["cue"]), cfg["stage3"]["shortcut"]["offset_px"])
    cams = z["cam4"][rows][pick].astype(np.float32).reshape(-1, 8, 8)
    off, px = cfg["stage3"]["shortcut"]["offset_px"], spec["cue"]["cue_px"]

    fig, ax = plt.subplots(4, 6, figsize=(13, 9.4))
    for k, (i, kind) in enumerate(zip(pick, kinds)):
        for j, show_cam in enumerate((False, True)):
            a = ax[k // 3, (k % 3) * 2 + j]
            a.imshow(imgs[k])
            if show_cam:
                a.imshow(zoom(cams[k], 128 / 8, order=1), cmap="inferno", alpha=0.55, extent=(0, 128, 128, 0))
                a.set_title("layer4 Grad-CAM (8x8)", fontsize=7)
            else:
                a.set_title(f"{kind}; true: {CLASSES[lab[i]]}" + "\n" + f"pred: {CLASSES[pred[i]]} (p={conf[i]:.3f})", fontsize=7,
                            color="#D55E00" if kind != "correct" else "#0072B2")
            a.add_patch(plt.Rectangle((off, off), px, px, fill=False, ec="white", lw=1, ls="--"))
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Qualitative case study, reversed-cue shortcut v2 ({c['model']}); dashed square = pasted cue; selection: rng seed {c['seed']}\n"
                 "post-hoc / descriptive; images re-read from the local cache, CAMs are the saved maps", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(out_dir / "stage11" / "figures" / "fig7b_cam_cases.png", dpi=120)
    tab = pd.DataFrame({"kind": kinds, "uid": z["uid"][rows][pick], "true": [CLASSES[lab[i]] for i in pick], "pred": [CLASSES[pred[i]] for i in pick],
                        "confidence": conf[pick].round(4), "cam_mass_top_left_2x2_cells_l4": [float(m_[0:2, 0:2].sum()) for m_ in cams]})
    tab.to_csv(out_dir / "stage11" / "cam_cases.csv", index=False)
    print(tab.to_string())
