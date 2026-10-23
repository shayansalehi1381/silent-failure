"""Phase 11a (post-hoc / descriptive): population sensitivity - silent-failure detection AUROC / AUPRC / AURC among ACCEPTED rows only
(calibrated confidence >= accept_conf, the acceptance rule behind the sf_090 label) versus the primary all-rows population. Detectors and
fusion weights are exactly those of Phases 5 and 8 (asserted equal to Phase 5); only the evaluation population changes. No retraining."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from sfd import stage5 as s5
from sfd import stage7 as s7
from sfd import stage8 as s8

NAMES = ["M0", "M1", "M2", "sf_M1", "sf_M2"]
PAIRS = [("ACC", "M1", "M0", False), ("ACC", "M2", "M1", False), ("ACC", "sf_M2", "sf_M1", False), ("ACC", "sf_M1", "M1", False)]


def run(cfg: dict, out_dir: Path, log=print) -> None:
    c = cfg["stage11"]["accepted_only"]
    out = out_dir / "stage11"
    out.mkdir(parents=True, exist_ok=True)
    seeds, B, thr = cfg["train"]["seeds"], c["bootstrap_B"], c["accept_conf"]
    t0 = time.time()
    df = s5.load(cfg, out_dir)
    rows = []
    for H in ("A", "B", "C"):
        ctx = s8.make_ctx(df, H, "sf_090", seeds)
        sc, wts = s8.score_methods(ctx, NAMES, seeds)
        s7.check_weights_match_stage5({"heldout": H, "label": "sf_090", "weights": [w for w in wts if w["model"] in ("M0", "M1", "M2")]}, out_dir)
        masks = {s: (ctx["te"][s]["confidence"] >= thr).to_numpy() for s in seeds}
        te = {s: ctx["te"][s][masks[s]].reset_index(drop=True) for s in seeds}
        sub = {m: {s: v[masks[s]] for s, v in d.items()} for m, d in sc.items()}
        res = s8.evaluate(te, sub, "sf_090", seeds, B, refs=False)
        r = s8.make_rows(res, {"heldout": H, "label": "sf_090", "population": f"accepted rows only (confidence >= {thr})"}, PAIRS)
        r = [x for x in r if x.get("metric") in ("auroc", "auprc", "aurc", "dauroc", "dauprc", "daurc")]
        for x in r:
            x["accepted_share"] = float(sum(m.mean() for m in masks.values()) / len(seeds))
        rows += r
        log(f"[stage11] accepted-only held-out {H} done ({time.time() - t0:.0f} s)")
    pd.DataFrame(rows).to_csv(out / "accepted_only.csv", index=False)
    log(f"[stage11] all done ({time.time() - t0:.0f} s)")
