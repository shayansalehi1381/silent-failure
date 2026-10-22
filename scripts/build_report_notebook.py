"""Builds silent_failure_report.ipynb (the final deliverable) from saved, committed results only.

python scripts/build_report_notebook.py && jupyter nbconvert --to notebook --execute --inplace silent_failure_report.ipynb
The notebook reads small committed files under outputs/ and docs/ (no training, no GPU, no per-row arrays).
"""
from __future__ import annotations

from pathlib import Path

import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str) -> None:
    from reflow import reflow          # wrap long lines (no formatter installed); semantics unchanged
    cells.append(nbf.v4.new_code_cell(reflow(s.strip("\n"))))


# =====================================================================================================================
md(r"""
# Silent failure detection of medical image classifiers under distribution shift

**Question.** Do *representation drift* (where an image's features lie relative to the training data) and *Grad-CAM explanation
drift* (whether the model "looks" at an image the way it usually does) detect **silent failures** - wrong predictions made with high
calibrated confidence - beyond calibrated predictive uncertainty? And do they lower the residual silent-failure rate when a fixed share
of cases can be sent to a human?

This notebook is the single final report. It **reads saved results only** (small CSV / JSON / PNG / Markdown files committed under
`outputs/` and `docs/`), performs no training, no GPU work and no feature extraction, and runs top-to-bottom from a fresh clone in a few
minutes. The heavy pipeline that produced these files is in `sfd/` and `scripts/` (see Section 14).

## Contents
0. [How to read this notebook, glossary, executive summary, main findings](#s0)
1. [Problem and definitions](#s1)
2. [Data, splits and shift families](#s2)
3. [Methods](#s3)
4. [Classifier performance and calibration (Table 1)](#s4)
5. [Behaviour under shift (Figure 2)](#s5)
6. [Sample-level detection (Figures 3-4, Table 2)](#s6)
7. [Batch (window) monitoring (Figure 5)](#s7)
8. [Referral policy under a review budget (Figure 6, Table 3)](#s8)
9. [Ablations and sensitivity (Figure 8)](#s9)
10. [EXPLORATORY, POST-HOC: targeted (confidence-gated) fusion](#s10)
10b. [Extensions (post-hoc): deep-ensemble baseline and in-family vs out-of-family](#sx)
11. [Shortcut case study (Figure 7)](#s11)
12. [Blueprint compliance checklist and deviations](#s12)
13. [Honest limitations](#s13)
14. [Reproducibility](#s14)
- [Appendix A: all remaining tables](#sA)
- [Appendix B: file index](#sB)
""")

md(r"""
> **Known gaps (Blueprint items not done).** (1) a second classifier architecture; (2) MC-dropout uncertainty (a deep-ensemble baseline
> was added afterwards as a post-hoc extension, Section 10b); (3) Grad-CAM alternatives (Grad-CAM++, ScoreCAM, Integrated Gradients);
> (4) a within-family tuned in-family vs out-of-family comparison (only source-vs-shifted and in-sample evidence, Section 10b);
> (5) the artefact-patch (non-tissue) removal ablation in Camelyon. They are listed as DEVIATIONS in Section 12; no result in the
> main sections depends on them.
""")

code(r"""
# ---------------------------------------------------------------- setup: imports, style, helpers (defined once) ----
import json, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from IPython.display import display, Markdown, Image, HTML

warnings.filterwarnings("ignore")
T_START = time.time()
ROOT = Path.cwd()
OUT = ROOT / "outputs"
assert OUT.exists(), "run the notebook from the repository root"

# Okabe-Ito colour-blind-safe palette, one fixed colour per method so every figure reads the same way
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
COL = {"random": OI["grey"], "oracle": OI["black"], "M0": OI["blue"], "M1": OI["orange"], "M2": OI["green"],
       "sf_M1": OI["purple"], "sf_M2": OI["vermillion"], "G_M0": OI["sky"], "G_M1": "#882255", "G_sfM1": OI["purple"]}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.prop_cycle": plt.cycler(color=[OI[k] for k in ("blue", "orange", "green", "vermillion", "purple", "sky", "yellow", "black")])})
pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 250, "display.max_colwidth", 120)

_legend0 = plt.Axes.legend


def _legend_out(self, *a, **k):
    # Every legend sits BELOW its axes (outside the data area), lower still if the x tick labels are rotated.
    if "bbox_to_anchor" not in k and "handles" not in k:
        rot = any(t.get_rotation() > 0 for t in self.get_xticklabels())
        k.update(loc="upper center", bbox_to_anchor=(0.5, -0.42 if rot else -0.2), frameon=False)
        k.setdefault("ncol", 3)
    return _legend0(self, *a, **k)


plt.Axes.legend = _legend_out

READ = []                                                   # every file read, for Appendix B
DEFERRED = []                                               # long tables moved to Appendix A
IN_APPENDIX = False
RENAME = {"p_boot": "bootstrap p", "q_bh": "BH q", "jackknife_min": "leave-one-seed-out min", "jackknife_max": "leave-one-seed-out max",
          "heldout": "held-out family", "n_per_seed": "rows per seed", "sf_rate": "silent-failure prevalence", "prevalence": "SF prevalence",
          "c_kind": "shortcut model", "level_col": "level type", "kind": "row type", "comparator_value": "comparator value",
          "workload_flag": "off budget by > 1 pt", "workload_difference": "workload difference", "workload_matched": "workload matched",
          "boundary_tie_share": "boundary tie share", "n_windows": "windows", "n_groups": "groups", "access_licence": "licence",
          "hospital_counts": "patches per hospital", "class_counts": "images per class", "flagged_nontissue": "non-tissue flagged",
          "acc_mean": "accuracy (mean)", "ece_mean": "ECE (mean)", "T_standard": "temperature T", "stat": "statistic", "H": "family"}


def rd(rel):
    # Load one committed result file (csv / json / parquet / md) and remember it for the file index.
    p = OUT / rel if not rel.startswith(("docs/", "configs/")) else ROOT / rel
    READ.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return p.read_text(encoding="utf-8")


def ci(e, lo, hi, d=3, scale=1.0, signed=False):
    # 'estimate [low, high]' string; scale=100 turns rates into percent / percentage points.
    if pd.isna(e):
        return ""
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    s = f.format(scale * e)
    return s if pd.isna(lo) else f"{s} [{f.format(scale * lo)}, {f.format(scale * hi)}]"


def fmt(df, est="estimate", lo="ci_low", hi="ci_high", d=3, scale=1.0, signed=False, keep=None, name="estimate [95% CI]"):
    # Table with one formatted 'estimate [95% CI]' column instead of three numeric ones.
    out = df.copy()
    out[name] = [ci(e, l, h, d, scale, signed) for e, l, h in zip(out[est], out[lo], out[hi])]
    cols = [c for c in (keep or [c for c in df.columns if c not in (est, lo, hi)]) if c in out.columns] + [name]
    return out[cols].reset_index(drop=True)


def show(df, title, keep=False):
    # Display a table with a bold title. Tables with more than 20 rows go to Appendix A (full, same data) unless keep=True.
    df = df.reset_index(drop=True).rename(columns=RENAME)
    if len(df) > 20 and not keep and not IN_APPENDIX:
        DEFERRED.append((title, df))
        display(Markdown(f"*Full table '{title}' ({len(df)} rows) is in Appendix A.D{len(DEFERRED)}; a compact version is shown above or in the text.*"))
        return
    display(Markdown(f"**{title}** - {len(df)} rows"))
    display(HTML(df.to_html(index=False, na_rep="", border=0, max_rows=None, max_cols=None)))


def say(text):
    display(Markdown(text))


def figcap(n, text, notice):
    say(f"*Figure {n}. {text}*  \n**What to notice:** {notice}")


def png(rel, width=None):
    READ.append("outputs/" + rel)
    display(Image(filename=str(OUT / rel), width=width))


def one(df, **kw):
    # The single row matching every key=value (floats compared with a tolerance).
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= np.isclose(df[k].astype(float), v) if isinstance(v, float) else (df[k].astype(str) == str(v))
    r = df[m]
    assert len(r) == 1, (kw, len(r))
    return r.iloc[0]


def rc(r, d=3, scale=1.0, signed=True, est="estimate"):
    # Formatted 'estimate [CI]' of one table row.
    return ci(r[est], r["ci_low"], r["ci_high"], d, scale, signed)


print("setup ok; results folder:", OUT)
""")

# =====================================================================================================================
md(r"""
<a id="s0"></a>
## 0. How to read this notebook

* Every number in the text is **computed in the notebook from a saved table** and printed with its 95% confidence interval (CI).
  Rates are shown as proportions (0.123) or percent (12.3%); differences in residual rates are in **percentage points (pt)**.
* Every result is tagged with its evidential status:
  * **PRE-REGISTERED PRIMARY** - declared in a frozen plan committed (hash given) *before* any of its numbers were computed; no
    multiplicity adjustment.
  * **SECONDARY** - pre-registered secondary analyses or sensitivity analyses (Benjamini-Hochberg FDR within each declared family), and
    the outcome-informed but pre-committed amendment A1 of Phase 6.
  * **EXPLORATORY-POST-HOC** - motivated by earlier results; reported in clearly headed sections only, never in a main table, and never
    a replacement for the pre-registered detectors M0 / M1 / M2.
* Tables show **all rows** of the saved table they present (large tables are shown in full in Appendix A, with a filtered view in the
  main text that is labelled as such). Each table is followed by **What this shows**; each figure by a caption and **What to notice**.

### Glossary
| term | meaning |
|---|---|
| **Silent failure** | a wrong prediction whose *calibrated* confidence is high. Primary label `sf_090`: wrong AND confidence >= 0.90. |
| **MSP** | maximum softmax probability = the model's confidence; uncertainty = 1 - MSP. |
| **Temperature scaling** | dividing the logits by one scalar T fitted on source calibration data so that confidences match accuracies. |
| **ECE** | expected calibration error: average |accuracy - confidence| over confidence bins. |
| **AUROC** | probability that a random silent failure gets a higher monitor score than a random non-silent case (0.5 = chance). |
| **AUPRC** | area under the precision-recall curve (average precision); its chance level equals the prevalence. |
| **AURC** | area under the risk-coverage curve: accept cases from the lowest monitor score upwards and average the silent-failure rate among accepted cases (lower is better). |
| **FAR** | false-alarm rate of a batch monitor on windows that contain no shifted images (nominal 5%). |
| **CAM / Grad-CAM** | class activation map: a coarse heat map (8 x 8 at layer4) of image regions that drive the predicted class. |
| **JS divergence** | Jensen-Shannon divergence between two normalised maps (0 = identical). |
| **Mahalanobis / kNN** | distance of an image's 512-d penultimate features to the training-class Gaussians (Mahalanobis) or to its 10th nearest training neighbour (kNN). |
| **MMD** | maximum mean discrepancy: kernel two-sample statistic between a window's features and a training reference. |
| **M0 / M1 / M2** | nested detectors: M0 = uncertainty; M1 = M0 + Mahalanobis; M2 = M1 + CAM drift. |
| **Held-out-family rotation** | detector weights are fitted on two shift families and evaluated on the third one, so every result is *out-of-family*. |
| **Paired two-level cluster bootstrap** | resample the 5 classifier seeds, then clusters of near-identical images (PathMNIST) or patients (Camelyon); the same draw is used for every method, so differences are paired. 2,000 resamples. |
| **Pre-registered vs exploratory** | see the tags above. |
""")

code(r"""
# ---------------------------------------------------------------- headline tables used throughout -----------------
t2 = rd("stage5/table2.csv")                                  # Phase 5 Table 2
t3 = rd("stage7/table3_referral.csv")                         # Phase 7 Table 3 (referral part)
t10_ = rd("stage10/e1_ensemble_baseline.csv")                  # Phase 10 extension E1
c6 = rd("stage6/contrasts.csv"); c6a = rd("stage6/a1/contrasts.csv")
far6 = rd("stage6/far_table.csv"); far6a = rd("stage6/a1/far_table.csv")
tt = rd("stage7/threshold_transfer.csv")
G = {g: rd(f"stage8/tables/ablation_{g}.csv") for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9")}
EX = rd("stage8/tables/exploratory_targeted.csv")
ftau = rd("stage4/frozen_tau.csv")

def t2r(H, q, m): return one(t2, heldout=H, quantity=q, metric=m)
def t3r(H, q, m, b): return one(t3, heldout=H, quantity=q, metric=m, budget=float(b))
def g8(H, q, b, m="dresid_sf", g="G2"):
    d = G[g]; return one(d[(d.kind == "contrast") & (d.level_col.isna())], heldout=H, quantity=q, metric=m, budget=float(b))
def ex(H, q, b, m="dresid_sf"):
    return one(EX, heldout=H, quantity=q, metric=m, budget=float(b))
def c6r(tab, H, a, b, kind="standard"):
    return one(tab[(tab["set"] == "primary") & (tab["size"] == 100) & (tab["mask"] == "silent_pooled")], H=H, kind=kind, a=a, b=b)
def rc6(r): return ci(r["diff"], r["ci_low"], r["ci_high"], 3, 1, True)
def e20(H, q): return rc(one(t10_, kind="contrast", heldout=H, quantity=q, metric="dresid_sf", budget=0.2), 2, 100)
MARK = {"reduces": "v", "improves": "v", "increases": "^", "worse": "^", "no practical difference": "=", "inconclusive": "?"}

def ab_compact(d, quantities, budgets=(0.2,), level_col=None, rows_by=None, heldouts="ABC", metric="dresid_sf"):
    # Wide table: one row per contrast (and level), one column per family (and budget); cell = 'difference in pt [95% CI] mark'.
    # mark: v = reduces residual silent failures, ^ = increases, = no practical difference, ? inconclusive.
    d = d[(d.kind == "contrast") & (d.metric == metric) & d.quantity.isin(quantities)]
    d = d[d.level_col.isna()] if level_col is None else d[d.level_col == level_col]
    out = {}
    for _, r in d.iterrows():
        for b in budgets:
            if not np.isclose(r.budget, b):
                continue
            key = tuple([r.quantity] + [str(r[c]) for c in (rows_by or [])])
            col = f"{r.heldout} @{int(100 * b)}%" if len(budgets) > 1 else f"{r.heldout}"
            out.setdefault(key, {})[col] = f"{ci(r.estimate, r.ci_low, r.ci_high, 2, 100, True)} {MARK.get(r.verdict, '')}"
    idx = ["contrast"] + [{"label": "label", "level": "level", "c_kind": "shortcut model"}.get(c, c) for c in (rows_by or [])]
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=idx)
    order = [q for q in quantities if q in df.index.get_level_values(0)]
    df = df.reset_index()
    df["_o"] = df["contrast"].map({q: i for i, q in enumerate(order)})
    return df.sort_values(["_o"] + idx[1:], kind="stable").drop(columns="_o").reset_index(drop=True)

nfar_pre = int(far6[far6["size"] == 100].flag_outside_3_7pct.sum()); nfar_a1 = int(far6a[far6a["size"] == 100].flag_outside_3_7pct.sum())
ncell = int((far6["size"] == 100).sum())
n_tt = int(tt[tt.metric == "realised_workload"].workload_flag.astype(str).eq("True").sum()); N_tt = int((tt.metric == "realised_workload").sum())
n_degen = int(ftau.degenerate.notna().sum())
print("loaded")
""")

code(r"""
# ---------------------------------------------------------------- executive summary (generated from the numbers) ---
say(f'''
### Executive summary (plain language)
We trained image classifiers (ResNet-18, 5 random seeds each) on two pathology datasets and then fed them images that differ from the
training data in three ways: **(A)** artificially corrupted images (blur, brightness, JPEG, ...), **(B)** tissue from two hospitals the
model never saw, and **(C)** images carrying a coloured marker that the model learned as a shortcut, which we then removed or reversed.
We asked whether three kinds of warning signals catch the dangerous errors - wrong answers given with at least 90% confidence:
the model's own uncertainty (**M0**), the distance of the image's internal features from the training data (**M1** adds this),
and a change in the model's attention map (**M2** adds this). Detectors were always tuned on two kinds of shift and tested on the third.

* Compared with **single-model (one-seed) uncertainty**, feature distance helps on corrupted images and new hospitals but **hurts on the shortcut**: AUROC gain M1-M0 =
  {rc(t2r("A","M1-M0","dAUROC"))} (A), {rc(t2r("B","M1-M0","dAUROC"))} (B), {rc(t2r("C","M1-M0","dAUROC"))} (C).
* Against a 4-seed deep-ensemble baseline (post-hoc extension), M1 no longer beats the baseline ({e20('B','M1-E_fuse')} pt in B and {e20('C','M1-E_fuse')} pt in C
  worse at a 20% budget; A {e20('A','M1-E_fuse')} pt, inconclusive), while Mahalanobis added on top of the ensemble still helps on A and B
  ({e20('A','E_M1-E_fuse')}, {e20('B','E_M1-E_fuse')} pt) and hurts on C ({e20('C','E_M1-E_fuse')} pt).
* The attention-map signal, used the pre-registered way (it may only *raise* the alarm), adds **nothing**: M2-M1 =
  {rc(t2r("A","M2-M1","dAUROC"),4)} (A), {rc(t2r("B","M2-M1","dAUROC"),4)} (B), {rc(t2r("C","M2-M1","dAUROC"),4)} (C).
* If a hospital can review a fixed share of cases, ranking by uncertainty sends the *unsure* cases first - but silent failures are by
  definition *sure*. At a 5% review budget the uncertainty ranking leaves **more** silent failures than random review
  ({rc(t3r("A","M0-random","dresid_sf",0.05),2,100)} pt on A). Adding feature distance helps only at a 20% budget
  (A {rc(t3r("A","M1-M0","dresid_sf",0.2),2,100)} pt, B {rc(t3r("B","M1-M0","dresid_sf",0.2),2,100)} pt) and hurts on the shortcut
  (C {rc(t3r("C","M1-M0","dresid_sf",0.2),2,100)} pt).
* Batch monitors calibrated on one source split raised far too many false alarms on another source split ({nfar_pre} of {ncell}
  cells outside the 3-7% tolerance); this pre-registered check **failed** and is reported openly with a pre-committed fix (A1).
* Ablation finding (pre-registered): if the attention-map signal is allowed a **negative** weight, it reduces residual silent failures on
  A ({rc(g8("A","sf_M2-sf_M1",0.10),2,100)} pt at 10%) and C ({rc(g8("C","sf_M2-sf_M1",0.10),2,100)} pt) - confident errors tend to look
  *typical*, not unusual.
* Exploratory only: ranking within the confident cases first (confidence >= 0.90) cuts residual silent failures strongly on A and B
  (gated M1 vs M1 at 20%: A {rc(ex("A","G_M1-M1",0.2),2,100)} pt, B {rc(ex("B","G_M1-M1",0.2),2,100)} pt).
''')
""")

code(r"""
# ---------------------------------------------------------------- main findings table ----------------------------
P5, P6, A1, P7, P8 = "e388dd6", "931fb36", "4ea78c3", "aa962a9", "f0553df"
rows = [
 ("Representation drift (M1) improves out-of-family detection AUROC over single-model (one-seed) uncertainty on A and B", "PRE-REGISTERED PRIMARY", P5,
  f"A {rc(t2r('A','M1-M0','dAUROC'))}; B {rc(t2r('B','M1-M0','dAUROC'))}"),
 ("Representation drift hurts detection on the strong shortcut (C)", "PRE-REGISTERED PRIMARY", P5, f"C {rc(t2r('C','M1-M0','dAUROC'))}"),
 ("Explanation drift (non-negative M2) adds nothing over M1", "PRE-REGISTERED PRIMARY", P5,
  f"A {rc(t2r('A','M2-M1','dAUROC'),4)}; B {rc(t2r('B','M2-M1','dAUROC'),4)}; C {rc(t2r('C','M2-M1','dAUROC'),4)}"),
 ("Batch false-alarm check failed (val_cal thresholds do not transfer to id_test windows)", "PRE-REGISTERED PRIMARY", P6,
  f"{nfar_pre}/{ncell} cells outside 3-7% (after A1: {nfar_a1}/{ncell})"),
 ("At matched FAR, MMD beats mean uncertainty on silent windows only for new hospitals (B)", "SECONDARY (amendment A1)", A1,
  f"B {rc6(c6r(c6a,'B','r_mmd','u_mean'))}; A {rc6(c6r(c6a,'A','r_mmd','u_mean'))}"),
 ("Uncertainty-first referral leaves MORE silent failures than random at a 5% budget", "PRE-REGISTERED PRIMARY (reference contrast)", P7,
  f"M0-random: A {rc(t3r('A','M0-random','dresid_sf',0.05),2,100)} pt; B {rc(t3r('B','M0-random','dresid_sf',0.05),2,100)} pt"),
 ("Relative to single-model uncertainty (M0), M1 lowers residual silent failures only at a 20% budget (A, B) and raises them on C", "PRE-REGISTERED PRIMARY", P7,
  f"20%: A {rc(t3r('A','M1-M0','dresid_sf',0.2),2,100)}; B {rc(t3r('B','M1-M0','dresid_sf',0.2),2,100)}; C {rc(t3r('C','M1-M0','dresid_sf',0.2),2,100)} pt"),
 ("Blueprint-literal transferred thresholds miss the review budget", "SECONDARY", P7, f"{n_tt}/{N_tt} cells off by > 1 pt of workload"),
 ("Sign-free CAM drift (sf-M2) reduces residual silent failures on A and C", "PRE-REGISTERED PRIMARY (P2)", P8,
  f"10%: A {rc(g8('A','sf_M2-sf_M1',0.1),2,100)}; B {rc(g8('B','sf_M2-sf_M1',0.1),2,100)}; C {rc(g8('C','sf_M2-sf_M1',0.1),2,100)} pt"),
 ("Confidence-gated M1 ranking reduces residual silent failures vs plain M1", "EXPLORATORY-POST-HOC", P8,
  f"20%: A {rc(ex('A','G_M1-M1',0.2),2,100)}; B {rc(ex('B','G_M1-M1',0.2),2,100)}; C {rc(ex('C','G_M1-M1',0.2),2,100)} pt"),
 ("A leave-seed-out deep ensemble recovers most of M1's 20%-budget benefit over single-model uncertainty", "EXTENSION (post-hoc)", "6e6f05c",
  f"20%: E-fuse - M0 A {rc(one(t10_, kind='contrast', heldout='A', quantity='E_fuse-M0', metric='dresid_sf', budget=0.2),2,100)} pt; M1 - E-fuse A {rc(one(t10_, kind='contrast', heldout='A', quantity='M1-E_fuse', metric='dresid_sf', budget=0.2),2,100)} pt"),
]
MAIN = pd.DataFrame(rows, columns=["finding", "status", "plan commit", "key numbers (95% CI)"])
show(MAIN, "Main findings")
say("**What this shows:** the robust pre-registered results are mixed: relative to single-model (one-seed) uncertainty, representation drift helps for natural "
    "shifts but not for a dominant shortcut; explanation drift adds nothing as a monotone alarm; and uncertainty-first referral cannot reach silent failures at small "
    "budgets. **Explanation drift adds nothing as a monotone alarm (non-negative M2), but with a free sign it reduces residual silent failures on A and C "
    "(Phase 8, pre-registered); silent failures have LESS explanation drift than correct predictions (given M0 and Mahalanobis the logistic coefficient is negative); "
    "the mechanism is a hypothesis.** " + f"Against a 4-seed deep-ensemble baseline (post-hoc extension), M1 no longer beats the baseline (B {e20('B','M1-E_fuse')} pt, "
    f"C {e20('C','M1-E_fuse')} pt worse at a 20% budget; A {e20('A','M1-E_fuse')} pt, inconclusive), while Mahalanobis added on top of the ensemble still helps on A and B "
    f"({e20('A','E_M1-E_fuse')}, {e20('B','E_M1-E_fuse')} pt) and hurts on C.")
""")

# =====================================================================================================================
md(r"""
<a id="s1"></a>
## 1. Problem and definitions

A classifier is *deployed* on data that may differ from its training data. Most errors under shift come with low confidence and are
easy to flag. The dangerous errors are **silent**: the model is wrong but confident. Confidence here is always the **calibrated**
maximum softmax probability, $\hat p = \max_k \operatorname{softmax}(z/T)_k$, with temperature $T$ fitted on source calibration data.

**Labels.** For an image with true class $y$, prediction $\hat y$ and calibrated confidence $\hat p$:

* **`sf_090` (primary):** $\hat y \neq y$ and $\hat p \ge 0.90$.
* `sf_095` (sensitivity): $\hat y \neq y$ and $\hat p \ge 0.95$.
* `sf_tau` (Blueprint rule, sensitivity): $\hat y \neq y$ and $\hat p \ge \tau$, where $\tau$ is the smallest confidence threshold with
  $\ge 95\%$ precision (accuracy among accepted cases) on the source calibration split.
* `sf_099` (sensitivity): the same rule with a 99% precision target.

**Worked example (illustrative numbers, not data).** Five deployed images:
""")

code(r"""
ex_cases = pd.DataFrame({"image": [1, 2, 3, 4, 5], "true": ["tumour", "normal", "tumour", "normal", "tumour"],
                         "predicted": ["tumour", "tumour", "normal", "tumour", "normal"], "calibrated confidence": [0.98, 0.97, 0.62, 0.91, 0.93]})
ex_cases["wrong"] = ex_cases.true != ex_cases.predicted
ex_cases["sf_090"] = ex_cases.wrong & (ex_cases["calibrated confidence"] >= 0.90)
ex_cases["sf_095"] = ex_cases.wrong & (ex_cases["calibrated confidence"] >= 0.95)
ex_cases["uncertainty 1-MSP"] = 1 - ex_cases["calibrated confidence"]
show(ex_cases, "Illustrative cases (made-up numbers)")
say("**What this shows:** images 2, 4 and 5 are silent failures under `sf_090` (only image 2 under `sf_095`); image 3 is a *visible* "
    "error (confidence 0.62). An uncertainty ranking (largest 1-MSP first) refers image 3 first - a visible error - and ranks the "
    "silent failures 2, 4, 5 below it: with a budget of one case, uncertainty-first referral removes no silent failure. This is the "
    "structural problem quantified in Section 8.")

show(fmt(ftau.assign(estimate=ftau.tau, ci_low=np.nan, ci_high=np.nan), d=4, name="tau",
         keep=["model", "dataset", "kind", "seed", "val_cal_acc", "val_cal_precision_at_tau", "accept_frac", "degenerate"]),
     "Frozen Blueprint threshold tau per classifier (outputs/stage4/frozen_tau.csv)")
say(f"**What this shows:** for {n_degen} of {len(ftau)} classifiers the source calibration accuracy is already >= 0.95, so the "
    "95%-precision rule accepts *every* calibration prediction and tau falls to the lowest observed confidence; `sf_tau` then labels "
    "almost **every error** a silent failure. That label no longer separates silent from visible errors, so the fixed threshold 0.90 "
    "(`sf_090`) was made primary before any detection result (plan `docs/analysis_plan_frozen.md`, commit e388dd6); `sf_tau` and "
    "`sf_099` remain sensitivities.")
""")

# =====================================================================================================================
md(r"""
<a id="s2"></a>
## 2. Data, splits and shift families

* **PathMNIST** (MedMNIST v2, 128 x 128 px, 9 colorectal tissue classes; NCT-CRC-HE-100K). The official train + val images form one
  pool split into `train` / `val_es` (early stopping) / `val_cal` (calibration and monitoring references) / `id_test` with a
  **near-duplicate-cluster split** (clusters of near-identical images never cross splits). The official test set (CRC-VAL-HE-7K, a
  different clinical centre) is the supplement family **X** (`nct7k_test`).
* **Camelyon17-WILDS v1.0** (96 x 96 px lymph-node patches, tumour vs normal; 5 hospitals). Source hospitals 0, 3, 4; held-out
  hospitals 1 (`ood_c1`) and 2 (`ood_c2`) form family **B**. Splits are **patient-disjoint** (Phase 1b "Rank 1" assignment). Data were
  read from the unofficial Hugging Face mirror `wltjr1007/Camelyon17-WILDS` (counts verified against WILDS v1.0).
* **Family A** - MedMNIST-C v0.1.0: 11 corruptions x severities 1-5 applied at 224 px and resized to 128 px, plus the clean images =
  **56 conditions** on one fixed 3,000-image `id_test` sample.
* **Family C** - synthetic shortcut: a class-coded coloured cue in the image corner is present in training; at test it is `aligned`,
  `removed` or `reversed`. **v2** (cue on 100% of training images, 16 px; primary) and **v1** (95%; supplement), 5 seeds each.
""")

code(r"""
tab1 = rd("stage1/table1.csv")
show(tab1, "Table S1 - datasets, splits, roles, sizes, licences (outputs/stage1/table1.csv)")
say(f"**What this shows:** PathMNIST train has {int(one(tab1, dataset='pathmnist', split='train').n):,} images; Camelyon train "
    f"{int(one(tab1, dataset='camelyon17', split='train').n):,} patches from only "
    f"{int(one(tab1, dataset='camelyon17', split='train').n_groups)} patients; each held-out hospital contributes 8,000 patches.")

sp = rd("stage1/splits.parquet")
cam = sp[sp.dataset == "camelyon17"]
per_h = cam.groupby(["split", "hospital"]).agg(patches=("uid", "size"), patients=("patient", "nunique"),
                                               tumour_share=("label", "mean")).reset_index()
per_h["tumour_share"] = per_h.tumour_share.round(3)
show(per_h, "Camelyon17 patches, patients and tumour share per split and hospital (from outputs/stage1/splits.parquet)")
tr = cam[cam.split == "train"].label.mean(); src = cam[cam.split.isin(["val_es", "val_cal", "id_test"])].label.mean()
ood = cam[cam.split.isin(["ood_c1", "ood_c2"])].label.mean()
say(f"**What this shows:** the patient-level split leaves a **tumour-prior gap**: train is balanced ({tr:.1%} tumour) while the "
    f"other source splits ({src:.1%}) were matched to the held-out hospitals ({ood:.1%}). Cells are uneven (e.g. hospital 3 has a "
    f"single id_test patient). These are limitations of 10 / 4 / 6 / 6 source patients per split (Section 13).")

pm = sp[sp.dataset == "pathmnist"].groupby("split").agg(images=("uid", "size"), clusters=("group", "nunique")).reset_index()
show(pm, "PathMNIST images and near-duplicate clusters per split (from outputs/stage1/splits.parquet)")
say("**What this shows:** every PathMNIST split contains several thousand independent clusters; clusters are the resampling unit of "
    "the bootstrap because PathMNIST has no patient identifiers.")

b1 = rd("stage1b/best_candidate_detail.csv")
show(b1, "Phase 1b selected patient assignment (Rank 1) per split and hospital (outputs/stage1b/best_candidate_detail.csv)")
say("**What this shows:** each source split is drawn from all three source hospitals with at least 10% of its patches per hospital, "
    "but from only 1-5 patients per hospital.")
""")

code(r"""
ded = rd("stage1/dedup_report.csv")
show(ded, "Near-duplicate audit (outputs/stage1/dedup_report.csv)")
la = rd("stage1/leakage_audit.json")
chk = pd.DataFrame(la["checks"]); chk["value"] = chk["value"].astype(str).str.slice(0, 100)
show(chk, "Automated leakage checks (outputs/stage1/leakage_audit.json)")
say(f"**What this shows:** {int(chk.passed.sum())} of {len(chk)} leakage checks pass (no uid, near-duplicate cluster, patient or slide "
    f"crosses splits; held-out hospitals never appear in source splits).")
png("stage1/near_duplicate_pairs.png", width=700)
figcap("S1", "Example near-duplicate pairs found by the audit (PathMNIST).", "near-identical tiles exist inside the official pool; the cluster split keeps each cluster in one split.")
""")

code(r"""
cs = rd("stage3/condition_summary.csv")
show(fmt(cs.assign(estimate=cs.acc_mean, ci_low=np.nan, ci_high=np.nan), d=4, name="accuracy (mean over seeds)",
         keep=["family", "condition", "severity", "model_kind", "dataset", "n_seeds", "n", "acc_std", "conf_mean", "ece_mean", "sf_rate_mean", "sf_rate_std", "sf_among_errors_mean"]),
     "All evaluation conditions with per-condition behaviour (outputs/stage3/condition_summary.csv)")
say(f"**What this shows:** {int((cs.family == 'A').sum())} family-A rows (11 corruptions x 5 severities + clean), the Camelyon "
    "id_test and two held-out hospitals, three cue states for each shortcut model and the X supplement. These are the units over which "
    "the evaluation pools of Sections 6-10 are built.")
png("stage3/figures/examples.png", width=850)
figcap("S2", "Example images per shift family (corruptions at increasing severity, held-out hospitals, shortcut cue states).",
       "family A corruptions range from subtle (bubble, pixelate) to severe (brightness, contrast); the shortcut cue is a small corner patch.")
""")

# =====================================================================================================================
md(r"""
<a id="s3"></a>
## 3. Methods

**Classifier.** ResNet-18 initialised from ImageNet (torchvision `IMAGENET1K_V1`), SGD (lr 0.01, cosine, weight decay 5e-4), at most
8 epochs with early stopping on source `val_es` loss (patience 2), batch 128, mixed precision; augmentation = random flips and 90-degree
rotations only (no colour or blur augmentation, which would pre-train robustness to family A). 5 seeds per model kind (PathMNIST
standard, Camelyon standard, shortcut v1, shortcut v2) = 20 frozen classifiers.

**Temperature scaling** on `val_cal`: $T^\* = \arg\min_T \sum_i -\log \operatorname{softmax}(z_i/T)_{y_i}$.
**ECE** $= \sum_b \frac{|B_b|}{n} \lvert \operatorname{acc}(B_b) - \operatorname{conf}(B_b) \rvert$ over 15 equal-width confidence bins.

**Per-image signals** (higher = more suspicious):
* **M0 - uncertainty:** $u_1 = \log(1-\hat p)$ and $u_2 = \log H(\hat{\mathbf p})$, computed stably in log space from $z/T$ (identical
  ranking to $1-\hat p$ and entropy, but without float ties for saturated confidences). Energy is a descriptive extra.
* **M1 - representation:** Mahalanobis distance to the predicted class,
  $d_M(x) = \sqrt{(f(x)-\mu_{\hat y})^\top \Sigma^{-1} (f(x)-\mu_{\hat y})}$ with class means $\mu_c$ and one shared Ledoit-Wolf
  covariance $\Sigma$ fitted on penultimate features $f$ of **source training images only**, after the classifier was frozen
  (20,000 PathMNIST / 40,000 Camelyon reference images; frozen with checksums before any shifted file was read). kNN distance
  (10th neighbour, L2-normalised features) is a sensitivity.
* **M2 - explanation:** Grad-CAM at layer4 (8 x 8), normalised to unit mass; drift = Jensen-Shannon divergence to the predicted
  class's mean training CAM, divided by its typical variability under benign transforms (flips, rotations, +-10% brightness):
  $e(x) = \mathrm{JS}(\mathrm{CAM}(x), \overline{\mathrm{CAM}}_{\hat y}) / \mathrm{median\ transform\ JS}$. Layer3, layer2 (16 x 16),
  raw JS and a PCA-embedding distance are sensitivities.

**Standardisation.** Each signal is mapped to its percentile among the source `val_cal` values of the same classifier, with smooth
monotone tails outside the observed range (no shifted data are used).

**Fusion.** Logistic regression on the standardised signals, $s(x) = b_0 + \sum_j w_j\,x_j$, with **non-negative weights** $w_j \ge 0$
(the pre-registered monotone alarm), L2 penalty $10^{-3}$, equal weight for the two tuning families. The **sign-free** variant (Phase 8)
removes $w_j \ge 0$. **Held-out-family rotation:** weights are fitted per seed on two of A / B / C (label `sf_090`) and applied to the
third; X uses weights fitted on A + B + C.

**Sample-level endpoints.** AUROC, AUPRC and AURC $=\frac{1}{n}\sum_{k=1}^{n} \mathrm{risk}(k)$, where risk(k) is the silent-failure
rate among the $k$ accepted cases with the lowest score.

**Inference.** Paired two-level cluster bootstrap (B = 2,000, seed 0): resample the 5 classifier seeds with replacement, then clusters
(PathMNIST near-duplicate clusters; Camelyon patients) with replacement; the same draw is used for all methods. 95% percentile CIs;
$p = 2\min(P(d\le0), P(d\ge0))$; leave-one-seed-out jackknife. No adjustment for declared primaries; BH-FDR inside secondary families.

**Batch (window) monitoring.** Windows of $n=100$ images (sensitivities 50, 200) with a shifted fraction
$f \in \{0, 0.1, 0.25, 0.5, 0.75, 1\}$. Statistics: mean and 90th percentile of $\log(1-\hat p)$, the KS distance to source
uncertainties, unbiased RBF-kernel $\mathrm{MMD}^2$ between window features and a 2,000-image training reference, and the JS divergence
between the window's mean CAM and the class-mixture of training mean CAMs; plus a non-negative fused **combined monitor**. Alarm
threshold = 95th percentile over 3,000 source-only `val_cal` windows (5% nominal FAR); the **actual FAR** is measured on 2,000 held-out
`id_test` windows. A window is **silently failed** when its error rate exceeds the 95th percentile of source windows while its mean
confidence stays inside the source 2.5-97.5% range. **Detection delay** = number of post-onset windows until the first alarm,
right-censored at 30 (Kaplan-Meier median with censored share).

**Referral policy.** Exact budget $B \in \{5, 10, 20\}\%$: the top $B$ share of the held-out pool by score leaves automatic acceptance
(top $B/2$ = *refer*, next $B/2$ = *review*); ties at the boundary are split fractionally so workload is exactly $B$. Endpoint =
residual silent-failure rate among accepted cases; references: random referral (residual = prevalence) and an oracle (silent failures
first). Secondary, Blueprint-literal protocol: score thresholds tuned on the two tuning families and transferred, with the realised
workload reported.
""")

code(r"""
# ---------------------------------------------------------------- Figure 1: pipeline schematic --------------------
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_axis_off(); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
def box(x, y, w, h, text, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="black", lw=0.8, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.0))
box(0.1, 3.3, 2.0, 1.0, "Data + frozen splits\nPathMNIST, Camelyon17\nleakage audit (Ph. 1)", "#DDEEFF")
box(0.1, 1.8, 2.0, 1.0, "ResNet-18 x 5 seeds\n+ temperature scaling\n(Ph. 2; shortcut v1/v2 Ph. 3)", "#DDEEFF")
box(0.1, 0.3, 2.0, 1.0, "Shift families\nA corruptions, B hospitals\nC shortcut, X centre (Ph. 3)", "#FFF2CC")
box(2.8, 1.8, 2.2, 1.0, "Train-only references\nMahalanobis / kNN\nmean CAMs (Ph. 4)", "#E2F0D9")
box(2.8, 0.3, 2.2, 1.0, "Per-image signals\nM0 uncertainty, M1 +repr.\nM2 +CAM drift (Ph. 4)", "#E2F0D9")
box(5.7, 3.3, 2.6, 1.0, "Sample level (Ph. 5)\nfusion tuned on 2 families,\ntested on the 3rd: AUROC..", "#FCE4D6")
box(5.7, 1.8, 2.6, 1.0, "Batch level (Ph. 6)\nwindows, 5% FAR, power,\ndelay, silent windows", "#FCE4D6")
box(5.7, 0.3, 2.6, 1.0, "Referral (Ph. 7)\nexact review budget\n5/10/20%: residual SF rate", "#FCE4D6")
box(9.0, 1.8, 2.8, 1.0, "Ablations (Ph. 8)\nsign-free, layers, kNN, ref size,\nT, severity, cue, labels", "#EDE7F6")
box(9.0, 0.3, 2.8, 1.0, "EXPLORATORY, post-hoc\nconfidence-gated ranking", "#EEEEEE")
box(9.0, 3.3, 2.8, 1.0, "Frozen plans (hash) before\nevery analysis phase;\npaired cluster bootstrap", "#FFFFFF")
for a in [(1.1, 3.3, 1.1, 2.8), (1.1, 1.8, 1.1, 1.3), (2.1, 2.3, 2.8, 2.3), (2.1, 0.8, 2.8, 0.8), (3.9, 1.8, 3.9, 1.3),
          (5.0, 0.8, 5.7, 3.6), (5.0, 0.8, 5.7, 2.3), (5.0, 0.8, 5.7, 0.8), (8.3, 2.3, 9.0, 2.3), (8.3, 0.8, 9.0, 0.8)]:
    arrow(*a)
plt.show()
figcap(1, "Study pipeline. Phases (Ph.) are separate scripts; every analysis phase was pre-registered in a frozen plan before its numbers were computed.",
       "references and calibration use source data only; every detector is evaluated on a shift family it was not tuned on.")
""")

# =====================================================================================================================
md(r"""
<a id="s4"></a>
## 4. Classifier performance and calibration (Table 1)
Per-seed metrics on every split, before (`uncalibrated`) and after (`T_standard`) temperature scaling.
""")

code(r"""
mps = pd.concat([rd("stage2/metrics_per_seed_pathmnist.csv"), rd("stage2/metrics_per_seed_camelyon17.csv")], ignore_index=True)
for c in ("acc", "auroc", "nll", "ece", "brier", "T"):
    mps[c] = mps[c].round(4)
show(mps, "Table 1 - classifier accuracy, AUROC, NLL, ECE, Brier per dataset, seed, split and calibration variant")
summ = mps[mps.variant == "T_standard"].groupby(["dataset", "split"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
                                                                           ece_mean=("ece", "mean"), ece_sd=("ece", "std")).round(4).reset_index()
show(summ, "Table 1 summary - mean and SD over 5 seeds (calibrated)")
pmid = one(summ, dataset="pathmnist", split="id_test"); cmid = one(summ, dataset="camelyon17", split="id_test")
c2 = one(summ, dataset="camelyon17", split="ood_c2")
say(f"**What this shows:** PathMNIST is almost solved in distribution (id_test accuracy {pmid.acc_mean:.4f} +- {pmid.acc_sd:.4f}, ECE "
    f"{pmid.ece_mean:.4f}); Camelyon is harder and seed-dependent (id_test {cmid.acc_mean:.3f} +- {cmid.acc_sd:.3f}) and degrades on "
    f"hospital 2 (accuracy {c2.acc_mean:.3f}, ECE {c2.ece_mean:.3f}): the held-out hospitals are a real shift.")
cal = pd.concat([rd("stage2/calibration_pathmnist.csv"), rd("stage2/calibration_camelyon17.csv")], ignore_index=True)
show(cal, "Temperatures and training length per seed (outputs/stage2/calibration_*.csv)")
say(f"**What this shows:** fitted temperatures are {cal[cal.dataset=='pathmnist'].T_standard.min():.2f}-"
    f"{cal[cal.dataset=='pathmnist'].T_standard.max():.2f} (PathMNIST) and {cal[cal.dataset=='camelyon17'].T_standard.min():.2f}-"
    f"{cal[cal.dataset=='camelyon17'].T_standard.max():.2f} (Camelyon); Camelyon early-stopped after 2-3 epochs.")
rb = rd("stage2/reliability_bins.csv")
combos = [(d, v) for d in ("pathmnist", "camelyon17") for v in ("before temperature scaling", "after temperature scaling")]
fig, ax = plt.subplots(2, 4, figsize=(16, 5.8), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
gaps = []
for j, (ds, var) in enumerate(combos):
    g = rb[(rb.dataset == ds) & (rb.variant == var)]
    ax[0, j].plot([0, 1], [0, 1], "k--", lw=1)
    for seed, gs in g.groupby("seed"):
        k = gs[gs["count"] >= 30]                                    # rule: bins with fewer than 30 samples are not drawn
        ax[0, j].plot(k.bin_conf, k.bin_acc, "o-", ms=3, lw=1, label=f"seed {seed}")
    kept = g[g["count"] >= 30]
    gaps.append({"dataset": ds, "variant": var, "weighted mean |acc - conf|": (kept["count"] * (kept.bin_acc - kept.bin_conf).abs()).sum() / kept["count"].sum(),
                 "weighted mean (conf - acc)": (kept["count"] * (kept.bin_conf - kept.bin_acc)).sum() / kept["count"].sum(),
                 "share of images in dropped bins": 1 - kept["count"].sum() / g["count"].sum()})
    ax[0, j].set_title(f"{ds} id_test, {var}", fontsize=9); ax[0, j].set_xlim(0.45, 1.0); ax[0, j].set_ylim(0.3, 1.02); ax[0, j].set_ylabel("accuracy" if j == 0 else "")
    m = g.groupby("bin")["count"].mean()
    ax[1, j].bar(0.05 + 0.1 * m.index, m.values.clip(min=0.5), width=0.085, color=OI["grey"])
    ax[1, j].axhline(30, color=OI["vermillion"], ls="--", lw=1); ax[1, j].set_yscale("log"); ax[1, j].set_xlabel("confidence")
    ax[1, j].set_ylabel("images per bin (mean/seed)" if j == 0 else "")
ax[0, 0].legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(0.02, 0.98))
plt.tight_layout(); plt.show()
gp = pd.DataFrame(gaps)
g_ = lambda ds, v, c: float(gp[(gp.dataset == ds) & gp.variant.str.startswith(v)][c].iloc[0])
figcap("S3", "Reliability diagrams on id_test (10 equal-width confidence bins, one line per seed; bins with fewer than 30 images are not drawn; lower panels: images per "
             "bin on a log scale, red line = 30).",
       f"PathMNIST is well calibrated where the data are dense (the top bin holds almost all images; weighted |accuracy - confidence| {g_('pathmnist','before','weighted mean |acc - conf|'):.4f} before, "
       f"{g_('pathmnist','after','weighted mean |acc - conf|'):.4f} after scaling), and its lower-confidence bins are sparse and mostly dropped. Camelyon is over-confident before scaling "
       f"(mean confidence minus accuracy {g_('camelyon17','before','weighted mean (conf - acc)'):+.3f}) and only modestly less so after "
       f"({g_('camelyon17','after','weighted mean (conf - acc)'):+.3f}); single seeds still deviate visibly.")
show(gp.round(4), "Reliability summary behind Figure S3 (computed from outputs/stage2/reliability_bins.csv)")
""")

# =====================================================================================================================
md(r"""
<a id="s5"></a>
## 5. Behaviour under shift (Figure 2)
Accuracy, mean calibrated confidence and the silent-failure rate (`sf_090`) per condition, mean over 5 seeds (seed SD in the tables).
""")

code(r"""
fs = rd("stage3/family_summary.csv"); pca = rd("stage3/per_corruption_accuracy.csv"); psf = rd("stage3/per_corruption_sf_rate.csv")
ss = rd("stage3/shortcut_states.csv")
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
a = fs[(fs.family == "A")].sort_values("severity")
for m, lab, c in (("acc", "accuracy", OI["blue"]), ("conf", "mean confidence", OI["orange"]), ("sf_rate", "silent-failure rate", OI["vermillion"])):
    ax[0].errorbar(a.severity, a[m], yerr=a[m + "_std"], marker="o", ms=3, color=c, label=lab, capsize=2)
ax[0].set_xlabel("MedMNIST-C severity (0 = clean)"); ax[0].set_title("(a) family A, all corruptions"); ax[0].legend()
mat = psf.set_index("condition")[[str(s) for s in range(1, 6)]].drop(index="clean")
im = ax[1].imshow(mat.values, cmap="viridis", aspect="auto", vmin=0)
ax[1].set_yticks(range(len(mat))); ax[1].set_yticklabels(mat.index, fontsize=7); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(range(1, 6))
ax[1].set_xlabel("severity"); ax[1].set_title("(b) silent-failure rate per corruption"); plt.colorbar(im, ax=ax[1], fraction=0.046)
b = fs[(fs.dataset == "camelyon17")].set_index("condition").loc[["id_test", "ood_c1", "ood_c2"]]
x = np.arange(3); w = 0.27
for i, (m, lab, c) in enumerate((("acc", "accuracy", OI["blue"]), ("conf", "confidence", OI["orange"]), ("sf_rate", "SF rate", OI["vermillion"]))):
    ax[2].bar(x + (i - 1) * w, b[m], w, yerr=b[m + "_std"], color=c, label=lab, capsize=2)
ax[2].set_xticks(x); ax[2].set_xticklabels(["id_test (h 0,3,4)", "ood_c1 (h 1)", "ood_c2 (h 2)"], fontsize=7); ax[2].set_title("(c) family B, Camelyon"); ax[2].legend()
for i, (k, lab) in enumerate((("shortcut", "v1"), ("shortcutv2", "v2"))):
    s = ss[ss.model_kind == k].set_index("condition").loc[["aligned", "removed", "reversed"]]
    ax[3].bar(np.arange(3) + (i - 0.5) * 0.38, s.sf_rate, 0.38, yerr=s.sf_rate_std, label=f"{lab} SF rate", color=[OI["sky"], OI["vermillion"]][i], capsize=2)
    ax[3].plot(np.arange(3) + (i - 0.5) * 0.38, s.acc, "k" + ["o", "s"][i], ms=4, label=f"{lab} accuracy")
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(["aligned", "removed", "reversed"]); ax[3].set_title("(d) family C, cue states"); ax[3].legend(fontsize=7)
plt.tight_layout(); plt.show()
a5 = one(fs, family="A", severity=5); r2 = one(ss, model_kind="shortcutv2", condition="reversed")
figcap(2, "Behaviour under shift: (a) family A by severity (error bars = seed SD); (b) silent-failure rate per corruption and severity; "
          "(c) family B per hospital; (d) family C cue states for the weak (v1) and strong (v2) shortcut.",
       f"confidence falls far more slowly than accuracy, so silent failures grow with severity (severity 5: accuracy {a5.acc:.3f}, "
       f"confidence {a5.conf:.3f}, SF rate {a5.sf_rate:.3f}); under a reversed v2 cue the model is almost always wrong yet confident "
       f"(accuracy {r2.acc:.3f}, SF rate {r2.sf_rate:.3f}).")
show(fs, "Family-level summary (outputs/stage3/family_summary.csv)")
show(pca, "Accuracy per corruption (rows) and severity (columns) (outputs/stage3/per_corruption_accuracy.csv)")
show(psf, "Silent-failure rate per corruption and severity (outputs/stage3/per_corruption_sf_rate.csv)")
show(ss, "Shortcut cue states (outputs/stage3/shortcut_states.csv)")
say(f"**What this shows:** corruptions differ widely (brightness_up severity 5 SF rate {float(one(psf, condition='brightness_up')['5']):.3f} vs "
    f"bubble {float(one(psf, condition='bubble')['5']):.3f}); hospital 2 is the harder hospital (SF rate {one(fs, condition='ood_c2').sf_rate:.3f} vs "
    f"{one(fs, condition='ood_c1').sf_rate:.3f}); the weak shortcut v1 barely fails (reversed SF rate "
    f"{one(ss, model_kind='shortcut', condition='reversed').sf_rate:.3f}), which is why v2 is the primary family C.")
""")

# =====================================================================================================================
md(r"""
<a id="s6"></a>
## 6. Sample-level detection (Phase 5; plan `docs/analysis_plan_frozen.md`, commit e388dd6)
Can a per-image score rank silent failures above other cases on a family it was not tuned on? Pools: A = clean + all 55 corrupted
conditions; B = Camelyon id_test + ood_c1 + ood_c2; C = aligned + removed + reversed (v2); X = PathMNIST id_test + nct7k_test.
""")

code(r"""
f3 = rd("stage5/fig3_data.csv")
sig_order = ["u_log1msp", "u_logent", "r_maha_pred", "r_knn", "e_js_norm_l4", "e_js_norm_l3", "e_js_norm_l2", "e_js_raw_l4", "e_pca_l4"]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
for a_, H in zip(ax, "ABCX"):
    d = f3[(f3.family == H) & (f3.metric == "AUROC")].set_index("signal").loc[sig_order]
    cols = [OI["blue"]] * 2 + [OI["orange"]] * 2 + [OI["green"]] * 5
    a_.barh(range(len(d)), d.estimate, xerr=[d.estimate - d.ci_low, d.ci_high - d.estimate], color=cols, capsize=2)
    a_.axvline(0.5, color="k", lw=0.7, ls="--"); a_.set_yticks(range(len(d))); a_.set_yticklabels(d.index if H == "A" else [], fontsize=8)
    a_.invert_yaxis(); a_.set_title(f"family {H}"); a_.set_xlabel("AUROC for sf_090")
plt.tight_layout(); plt.show()
figcap(3, "Single-signal detection AUROC of silent failures per family (blue = uncertainty, orange = representation, green = explanation; "
          "95% cluster-bootstrap CIs, 1,000 resamples, no fitting). SECONDARY (descriptive).",
       f"pooled C AUROCs look moderate (log(1-MSP) {one(f3, family='C', signal='u_log1msp', metric='AUROC').estimate:.3f}, "
       f"Mahalanobis {one(f3, family='C', signal='r_maha_pred', metric='AUROC').estimate:.3f}) only because aligned and removed images are pooled "
       "with reversed ones; within the reversed state every signal is inverted (inversion diagnosis below).")
show(fmt(f3, d=3, keep=["family", "signal", "metric"]), "Figure 3 data (outputs/stage5/fig3_data.csv)")
say(f"**What this shows:** on A, Mahalanobis alone reaches AUROC {rc(one(f3, family='A', signal='r_maha_pred', metric='AUROC'), signed=False)} vs "
    f"{rc(one(f3, family='A', signal='u_log1msp', metric='AUROC'), signed=False)} for log(1-MSP); the CAM signals stay near 0.5-0.6 everywhere.")
""")

code(r"""
# Figure 4 + Table 2 (PRE-REGISTERED PRIMARY)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for a_, met in zip(ax, ("dAUROC", "dAUPRC", "dAURC")):
    for j, q in enumerate(("M1-M0", "M2-M1")):
        d = t2[(t2.quantity == q) & (t2.metric == met)].set_index("heldout").loc[list("ABCX")]
        a_.errorbar(np.arange(4) + (j - 0.5) * 0.25, d.estimate, yerr=[d.estimate - d.ci_low, d.ci_high - d.estimate], fmt="o",
                    color=[OI["orange"], OI["green"]][j], capsize=3, label=q)
    a_.axhline(0, color="k", lw=0.7); a_.set_xticks(range(4)); a_.set_xticklabels(["A", "B", "C", "X (suppl.)"]); a_.set_title(met)
ax[0].legend(); plt.tight_layout(); plt.show()
figcap(4, "Incremental value of adding representation (M1-M0) and explanation drift (M2-M1), held-out family, 95% paired CIs "
          "(lower dAURC is better). PRE-REGISTERED PRIMARY.",
       "M1 helps on A and B and hurts on C; M2-M1 sits exactly on zero because the non-negative fusion gave the CAM weight 0.")
t2f = fmt(t2, d=4, signed=False, keep=["heldout", "analysis", "label", "quantity", "metric", "n_per_seed", "sf_rate", "p_boot", "jackknife_min", "jackknife_max", "verdict"])
def t2c():
    rows = []
    for H in "ABCX":
        for m in ("AUROC", "AUPRC", "AURC"):
            r = {"held-out family": H + (" (supplement)" if H == "X" else ""), "metric": m}
            for q in ("M0", "M1", "M2"):
                d = one(t2, heldout=H, quantity=q, metric=m); r[q] = ci(d.estimate, d.ci_low, d.ci_high, 3, 1, False)
            vs = []
            for q, lab in (("M1-M0", "M1 - M0 [95% CI]"), ("M2-M1", "M2 - M1 [95% CI]")):
                d = one(t2, heldout=H, quantity=q, metric="d" + m); r[lab] = ci(d.estimate, d.ci_low, d.ci_high, 4, 1, True); vs.append(d.verdict)
            r["verdict (M1-M0 / M2-M1)"] = " / ".join(vs)
            rows.append(r)
    return pd.DataFrame(rows)
show(t2c(), "Table 2 - sample-level detection of silent failures (sf_090), held-out-family rotation, mean over 5 seeds [95% CI]; AURC: lower is better")
show(t2f, "Table 2 (full, 60 rows) - M0 / M1 / M2 AUROC, AUPRC, AURC and primary contrasts (outputs/stage5/table2.csv)")
say(f"**What this shows:** primary contrasts (verdict rule: 'improves' only if the 95% CI excludes 0): M1-M0 dAUROC "
    f"A {rc(t2r('A','M1-M0','dAUROC'))} ({t2r('A','M1-M0','dAUROC').verdict}), B {rc(t2r('B','M1-M0','dAUROC'))} "
    f"({t2r('B','M1-M0','dAUROC').verdict}), C {rc(t2r('C','M1-M0','dAUROC'))} ({t2r('C','M1-M0','dAUROC').verdict}). "
    f"M2-M1 dAUROC is {t2r('A','M2-M1','dAUROC').estimate:+.1e} (A), {t2r('B','M2-M1','dAUROC').estimate:+.4f} (B), "
    f"{t2r('C','M2-M1','dAUROC').estimate:+.1e} (C): A's formal 'worse' verdict reflects a CI around -1e-6, i.e. no practical effect.")
fw = rd("stage5/fusion_weights.csv")
fwm = fw.groupby(["heldout", "model"])[[c for c in fw.columns if c.startswith("w_")]].mean().round(3).reset_index()
show(fwm, "Fusion weights (mean over seeds) per held-out family and detector (outputs/stage5/fusion_weights.csv; per-seed rows in Appendix A)")
say("**What this shows:** the CAM weight is exactly 0 when A, C or X is held out and small (about 0.11) when B is held out - the "
    "non-negative fusion cannot use CAM drift.")
il = rd("stage5/incremental_logit.csv")
show(il.round(4), "Incremental logistic regression on the held-out family (descriptive; outputs/stage5/incremental_logit.csv)")
say(f"**What this shows:** given M0 + Mahalanobis, the unconstrained CAM coefficient is negative in every held-out family (median "
    f"{il[il.heldout=='A'].coef_cam.median():.2f} A, {il[il.heldout=='B'].coef_cam.median():.2f} B, {il[il.heldout=='C'].coef_cam.median():.2f} C): "
    "confident errors have *typical-looking* explanations - the reason the pre-registered monotone M2 adds nothing (and the motivation of the "
    "Phase 8 sign-free ablation). The p-values ignore clustering and are descriptive.")
""")

code(r"""
sst = rd("stage5/seed_stability_per_seed.csv")
show(sst.round(4), "Seed stability of M0 / M1 / M2 per held-out family and seed (outputs/stage5/seed_stability_per_seed.csv)")
rng_ = sst.groupby(["heldout", "model"]).AUROC.agg(["min", "max"]).round(3)
say(f"**What this shows:** per-seed AUROC ranges are narrow, e.g. A M1 {rng_.loc[('A','M1'),'min']}-{rng_.loc[('A','M1'),'max']}, "
    f"C M0 {rng_.loc[('C','M0'),'min']}-{rng_.loc[('C','M0'),'max']}; the conclusions do not hinge on one seed.")
sens5 = rd("stage5/sensitivity.csv")
v = sens5[sens5.quantity.str.contains("-") & (sens5.metric == "dAUROC")]
vv = v.assign(c=[ci(e, l, h, 3, 1, True) for e, l, h in zip(v.estimate, v.ci_low, v.ci_high)], col=v.heldout + " " + v.quantity)
show(vv.pivot_table(index="variant", columns="col", values="c", aggfunc="first").reset_index(),
     "Phase 5 sensitivity analyses (compact): dAUROC of M1 - M0 and M2 - M1 per variant and held-out family [95% CI]")
show(fmt(v, d=4, signed=True, keep=["variant", "heldout", "quantity", "metric", "p_boot", "q_bh", "jackknife_min", "jackknife_max"]),
     "Phase 5 sensitivity analyses: dAUROC contrasts per variant (filtered view of outputs/stage5/sensitivity.csv; full table in Appendix A)")
mx = v[v.quantity == "M2-M1"]
say(f"**What this shows:** no sensitivity variant makes M2 beat M1 (largest M2-M1 dAUROC {mx.estimate.max():+.4f}); the M1-M0 gain on A "
    f"survives kNN ({rc(one(v, variant='kNN instead of Mahalanobis', heldout='A', quantity='M1-M0'))}) and sf_095 "
    f"({rc(one(v, variant='label sf_095', heldout='A', quantity='M1-M0'))}) and shrinks under the degenerate sf_tau label "
    f"({rc(one(v, variant='label sf_tau', heldout='A', quantity='M1-M0'))}).")
sb = rd("stage5/secondary_breakdowns.csv")
v = sb[sb.quantity.str.contains("-") & (sb.metric == "dAUROC")]
vb = v[v.analysis.isin(["A by severity", "B by condition", "C by condition"])]
vb = vb.assign(c=[ci(e, l, h, 3, 1, True) + (f" (q={q:.3f})" if pd.notna(q) else "") for e, l, h, q in zip(vb.estimate, vb.ci_low, vb.ci_high, vb.q_bh)])
show(vb.pivot_table(index=["analysis", "level"], columns="quantity", values="c", aggfunc="first").reset_index().rename(columns={"M1-M0": "dAUROC M1 - M0 [95% CI] (BH q)", "M2-M1": "dAUROC M2 - M1 [95% CI] (BH q)"}),
     "Phase 5 secondary breakdowns (compact): dAUROC by severity and condition; corruption types in Appendix A")
show(fmt(v, d=4, signed=True, keep=["analysis", "heldout", "level", "quantity", "n_per_seed", "sf_rate", "p_boot", "q_bh"]),
     "Phase 5 secondary breakdowns: dAUROC per corruption, severity and condition (filtered view of outputs/stage5/secondary_breakdowns.csv; full in Appendix A)")
say(f"**What this shows:** SECONDARY (BH within each breakdown). M1-M0 grows with severity on A (severity 1 "
    f"{rc(one(v, analysis='A by severity', level='1', quantity='M1-M0'))}, severity 5 {rc(one(v, analysis='A by severity', level='5', quantity='M1-M0'))}) "
    f"and is negative in C removed ({rc(one(v, analysis='C by condition', level='removed', quantity='M1-M0'))}).")
""")

code(r"""
inv = rd("stage5/inversion_diagnosis.json")
au = pd.DataFrame(inv["auroc_sf090"]).T.reset_index().rename(columns={"index": "cue state"})
show(au.round(4), "Family C (v2) inversion diagnosis: single-signal AUROC pooled and within each cue state (outputs/stage5/inversion_diagnosis.json)")
prof = pd.DataFrame({k: {kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in d.items()} for k, d in inv["reversed_profile"].items()}).T
show(prof.reset_index().rename(columns={"index": "group (reversed state)"}), "Profile of correct vs silently wrong predictions under the reversed cue")
say(f"**What this shows:** within the reversed state every signal is far below 0.5 (log(1-MSP) {inv['auroc_sf090']['reversed']['u_log1msp']:.3f}, "
    f"Mahalanobis {inv['auroc_sf090']['reversed']['r_maha_pred']:.3f}), so the inversion is **not** a pooling artefact "
    f"({inv['reading']}). The few correct predictions are almost all tumour-epithelium images whose tissue overrides the cue; they are the most "
    "atypical cases on every signal, whereas the confident errors look like normal members of the cue's class.")
""")

# =====================================================================================================================
md(r"""
<a id="s7"></a>
## 7. Batch (window) monitoring (Phase 6; plan `docs/batch_plan_frozen.md`, commit 931fb36; amendment A1, commit 4ea78c3)
Can a monitor that looks at a *window* of 100 deployed images raise an alarm when the window is silently failing? The pre-registered
FAR check is shown first, because it **failed**.
""")

code(r"""
def far_pivot(t):
    d = t[t["size"] == 100].copy(); d["cell"] = d.H + " / " + d.kind
    p = d.pivot(index="cell", columns="stat", values="far").round(3)
    f = d.pivot(index="cell", columns="stat", values="flag_outside_3_7pct")
    return p.astype(str) + np.where(f.astype(str) == "True", " *", "")
show(far_pivot(far6).reset_index(), "Actual FAR on held-out id_test windows, PRE-REGISTERED thresholds (val_cal); * = outside 3-7%")
show(far_pivot(far6a).reset_index(), "Actual FAR, amendment A1 (thresholds from a disjoint row-hash half of the ID pool); * = outside 3-7%")
say(f"**What this shows:** with the pre-registered val_cal thresholds {nfar_pre} of {ncell} (statistic x family) cells are outside "
    f"the 3-7% tolerance (e.g. mean uncertainty on A: {one(far6, H='A', kind='standard', size=100, stat='u_mean').far:.3f}; combined monitor on A: "
    f"{one(far6, H='A', kind='standard', size=100, stat='comb').far:.3f}). Diagnosis (before any power was inspected): fresh val_cal windows "
    "alarm at 4.6-6.2% (docs/batch_plan_amendment_1.md), so the pipeline is unbiased, but val_cal and id_test are not exchangeable at window level: a ~0.1 SD row-level "
    "difference becomes ~1 SD for a mean over 100 images. Powers at such different FARs cannot be ranked. Amendment A1 (committed before "
    f"computing it) draws thresholds and evaluation windows from two halves of the same ID pool; {nfar_a1}/{ncell} cells remain outside "
    f"(mean uncertainty A {one(far6a, H='A', kind='standard', size=100, stat='u_mean').far:.3f}, B {one(far6a, H='B', kind='standard', size=100, stat='u_mean').far:.3f}).")
""")

code(r"""
pp = rd("stage6/power_pooled.csv"); ppa = rd("stage6/a1/power_pooled.csv")
dl = rd("stage6/detection_delay.csv"); dla = rd("stage6/a1/detection_delay.csv")
stats_ = ["u_mean", "u_p90", "u_ks", "r_mmd", "e_cam_l4", "comb"]
fams = [("A", "standard"), ("B", "standard"), ("C", "shortcutv2"), ("X", "standard")]
fig, ax = plt.subplots(2, 4, figsize=(16, 6.4))
for j, (H, k) in enumerate(fams):
    for i, (tab, lab, c) in enumerate(((pp, "pre-registered", OI["grey"]), (ppa, "A1 (FAR-matched)", OI["blue"]))):
        d = tab[(tab.H == H) & (tab.kind == k) & (tab["size"] == 100) & (tab["mask"] == "silent_pooled")].set_index("stat").loc[stats_]
        ax[0, j].bar(np.arange(6) + (i - 0.5) * 0.38, d.power, 0.38, yerr=[d.power - d.ci_low, d.ci_high - d.power], color=c, label=lab, capsize=2)
    ax[0, j].set_xticks(range(6)); ax[0, j].set_xticklabels(stats_, rotation=35, fontsize=7); ax[0, j].set_title(f"{H} ({k}): power on silent windows")
    d = dla[(dla.H == H) & (dla.kind == k) & (dla.onset.astype(str) == "1.0")]
    for i, (cell, g) in enumerate(d.groupby("cell")):
        g = g.set_index("stat").reindex(stats_)
        ax[1, j].errorbar(np.arange(6) + i * 0.12, g.median_delay, yerr=[g.median_delay - g.ci_low, g.ci_high - g.median_delay], fmt="o", ms=3, capsize=2, label=cell)
    ax[1, j].set_xticks(range(6)); ax[1, j].set_xticklabels(stats_, rotation=35, fontsize=7); ax[1, j].set_title(f"{H}: median delay (A1, abrupt onset)")
    ax[1, j].set_ylabel("windows to first alarm"); ax[1, j].legend(fontsize=6)
ax[0, 0].legend(); plt.tight_layout(); plt.show()
figcap(5, "Batch monitoring. Top: power (share of silently failed windows that alarm) at nominal 5% FAR, pre-registered thresholds vs "
          "FAR-matched A1, window size 100, 95% CIs. Bottom: Kaplan-Meier median detection delay after an abrupt full shift (A1; missing "
          "points = median not reached within 30 windows).",
       "pre-registered powers are inflated wherever the FAR is inflated; at matched FAR (A1) MMD beats mean uncertainty only for the new "
       "hospitals (B), and window CAM drift beats MMD only on the shortcut, where MMD is blind.")
cmp = []
for H, k in fams:
    for a_, b_ in (("comb", "u_mean"), ("r_mmd", "u_mean"), ("e_cam_l4", "r_mmd")):
        r0, r1 = c6r(c6, H, a_, b_, k), c6r(c6a, H, a_, b_, k)
        cmp.append({"held-out": H, "model": k, "contrast": f"{a_} - {b_}", "pre-registered (unequal FAR)": rc6(r0) + f" {r0.verdict}",
                    "A1 FAR-matched": rc6(r1) + f" {r1.verdict}", "silent windows": int(r1.n_windows)})
cmp = pd.DataFrame(cmp)
show(cmp, "Declared primary contrasts (power on silent windows), pre-registered vs amendment A1, side by side")
say(f"**What this shows:** at matched FAR the combined monitor never beats mean uncertainty (A {rc6(c6r(c6a,'A','comb','u_mean'))}); MMD beats "
    f"mean uncertainty only on B ({rc6(c6r(c6a,'B','r_mmd','u_mean'))}); window CAM drift beats MMD only on C v2 "
    f"({rc6(c6r(c6a,'C','e_cam_l4','r_mmd','shortcutv2'))}), where MMD is blind. The pre-registered column ranks statistics at different FARs and must not be "
    "read as a comparison.")
""")

code(r"""
pc = rd("stage6/power_cells.csv"); pca1 = rd("stage6/a1/power_cells.csv")
def share(t):
    d = t[(t["size"] == 100) & ~t.cell.str.startswith("Acorr") & (t.fraction > 0)]
    return d.groupby(["H", "kind"]).apply(lambda g: pd.Series({"shifted windows": int(g.n_windows.sum()), "silent windows": int(g.n_silent.sum()),
                                                               "silent share": round(g.n_silent.sum() / g.n_windows.sum(), 4)})).reset_index()
sh = share(pc).merge(share(pca1), on=["H", "kind"], suffixes=(" (pre-reg.)", " (A1)"))
show(sh, "Share of shifted windows (f > 0, primary cells, size 100) that are silently failed (computed from outputs/stage6/*/power_cells.csv)")
say(f"**What this shows:** silently failed windows are rare except for the new hospitals (B: {one(sh, H='B', kind='standard')['silent share (A1)']:.1%} "
    f"under A1; share = silent windows / all windows with shifted fraction > 0, window size 100, summed over the primary cells and 5 seeds). **Plan-wording correction:** the plan expected uncertainty statistics to be blind to silent windows by construction; that "
    "holds for mean confidence, but the log-scale mean uncertainty still alarms on many silent windows because the silent definition constrains "
    "the mean confidence, not its tails (see the power figure).")
w = c6a[(c6a["set"] == "primary") & (c6a["mask"] == "silent_pooled")]
ww = w.assign(c=[ci(e, l, h, 3, 1, True) for e, l, h in zip(w["diff"], w.ci_low, w.ci_high)], contrast=w.H + " " + w.kind + ": " + w.a + " - " + w.b)
show(ww.pivot_table(index="contrast", columns="size", values="c", aggfunc="first").reset_index().rename(columns=lambda x: f"window size {x}" if isinstance(x, (int, np.integer)) else x),
     "Window-size sensitivity (compact): A1 power difference on silent windows by window size [95% CI]")
show(fmt(w, est="diff", d=3, signed=True, keep=["H", "kind", "size", "a", "b", "power_a", "power_b", "n_windows", "p_boot", "q_bh", "bh_family", "verdict"]),
     "Window-size sensitivity (50 / 100 / 200), A1 primary contrasts on silent windows (filtered view of outputs/stage6/a1/contrasts.csv)")
base = w[w["size"] == 100].set_index(["H", "kind", "a", "b"])
sig = base[(base.ci_low > 0) | (base.ci_high < 0)]
same = sum(int(np.sign(r["diff"]) == np.sign(base.loc[k, "diff"])) for k, r in w[w["size"] != 100].set_index(["H", "kind", "a", "b"]).iterrows() if k in sig.index)
nsz = sum(1 for k in w[w["size"] != 100].set_index(["H", "kind", "a", "b"]).index if k in sig.index)
say(f"**What this shows:** SECONDARY (BH within the window-size family). Of the {len(sig)} size-100 contrasts whose CI excludes 0, "
    f"{same} of {nsz} size-50 / size-200 counterparts keep the same sign; contrasts near zero can flip.")
d1 = dla[dla.onset.astype(str) == "1.0"]
dd1 = d1.assign(c=[f"{m:.0f} ({100 * c:.0f}% cens.)" for m, c in zip(d1.median_delay, d1.censored_share)])
show(dd1.pivot_table(index="cell", columns="stat", values="c", aggfunc="first").reset_index(),
     "Detection delay (compact): median windows to first alarm after an abrupt full shift (A1) and censored share")
show(d1.round(3), "Detection delay after an abrupt full shift, A1 (filtered view of outputs/stage6/a1/detection_delay.csv; all onsets in Appendix A)")
say(f"**What this shows:** {int((d1.median_delay == 1).sum())} of {len(d1)} (cell, statistic) medians equal one window; MMD on hospital 1 rarely alarms "
    f"(censored share {one(d1, cell='B|ood_c1', stat='r_mmd').censored_share:.0%}); layer4 CAM drift has median delays up to "
    f"{d1[d1.stat == 'e_cam_l4'].median_delay.max():.0f} windows and censored shares up to {d1[d1.stat == 'e_cam_l4'].censored_share.max():.0%}.")
""")

# =====================================================================================================================
md(r"""
<a id="s8"></a>
## 8. Referral policy under a fixed review budget (Phase 7; plan `docs/referral_plan_frozen.md`, commit aa962a9)
The top B% of each held-out pool by monitor score leaves automatic acceptance (exact budget, no threshold transfer). Endpoint: the
residual silent-failure rate among accepted cases. Random referral leaves the prevalence; the oracle (silent failures first) is a ceiling.
""")

code(r"""
f6 = rd("stage7/fig6_data.csv")
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for a_, H in zip(ax, "ABCX"):
    for m in ("random", "M0", "M1", "M2", "oracle"):
        g = f6[(f6.heldout == H) & (f6.quantity == m) & (f6.metric == "resid_sf")].sort_values("budget")
        a_.plot(100 * g.budget, 100 * g.estimate, "--" if m == "oracle" else "-o", ms=3, color=COL[m], label=m)
        a_.fill_between(100 * g.budget, 100 * g.ci_low, 100 * g.ci_high, color=COL[m], alpha=0.15, lw=0)
    for b in (5, 10, 20):
        a_.axvline(b, color="k", lw=0.4, ls=":")
    a_.set_title(f"held-out {H}"); a_.set_xlabel("review budget (% of cases)")
ax[0].set_ylabel("residual silent-failure rate (%)"); ax[0].legend(); plt.tight_layout(); plt.show()
figcap(6, "Residual silent-failure rate among accepted cases vs review budget (exact budget; 95% paired CIs). Dotted lines: the "
          "pre-registered budgets 5 / 10 / 20%. PRE-REGISTERED PRIMARY.",
       "on A and B the uncertainty-based curves first RISE above random (the budget is spent on visible, low-confidence errors) and only fall "
       "once the low-confidence cases are exhausted; M1 and M2 overlap; the oracle shows how much is achievable.")
t3f = fmt(t3, d=2, scale=100, keep=["heldout", "analysis", "budget", "quantity", "metric", "prevalence", "n_per_seed", "p_boot", "jackknife_min", "jackknife_max", "verdict"],
          name="estimate [95% CI] (% or pt)")
t3f["budget"] = (100 * t3f.budget).round(1)
def t3c():
    rows = []
    for H in "ABCX":
        for b in (0.05, 0.1, 0.2):
            sub = t3[(t3.heldout == H) & np.isclose(t3.budget, b) & (t3.metric == "resid_sf")]
            r = {"held-out family": H + (" (supplement)" if H == "X" else ""), "budget": f"{int(100 * b)}%", "SF prevalence (%)": f"{100 * sub.prevalence.iloc[0]:.2f}"}
            for q in ("random", "M0", "M1", "M2", "oracle"):
                d = sub[sub.quantity == q].iloc[0]; r[q + " residual SF (%)"] = ci(d.estimate, d.ci_low, d.ci_high, 2, 100, False)
            for q in ("M1-M0", "M2-M1"):
                d = one(t3, heldout=H, quantity=q, metric="dresid_sf", budget=float(b)); r[q.replace("-", " - ") + " [95% CI] (pt)"] = ci(d.estimate, d.ci_low, d.ci_high, 2, 100, True) + f" {MARK.get(d.verdict, '')}"
            rows.append(r)
    return pd.DataFrame(rows)
show(t3c(), "Table 3 (referral part, compact) - residual silent-failure rate among accepted cases at exact budgets (v reduces, ^ increases, = no practical difference, ? inconclusive)")
show(t3f, "Table 3 (referral part) - residual SF rate, residual error rate, SF capture, AURC and contrasts for every family, budget and method "
          "(outputs/stage7/table3_referral.csv; rates in %, differences in points)")
say(f"**What this shows:** M1-M0: A 10% {rc(t3r('A','M1-M0','dresid_sf',0.1),2,100)} pt, 20% {rc(t3r('A','M1-M0','dresid_sf',0.2),2,100)} pt; "
    f"B 20% {rc(t3r('B','M1-M0','dresid_sf',0.2),2,100)} pt; C 5/10/20% {t3r('C','M1-M0','dresid_sf',0.05).estimate*100:+.2f} / "
    f"{t3r('C','M1-M0','dresid_sf',0.1).estimate*100:+.2f} / {t3r('C','M1-M0','dresid_sf',0.2).estimate*100:+.2f} pt (increases). M2-M1 is "
    f"0.00 pt on A and C and at most {t3[(t3.quantity=='M2-M1')&(t3.metric=='dresid_sf')&(t3.heldout=='B')].estimate.max()*100:+.2f} pt on B. "
    f"M0 captures {t3r('A','M0','capture',0.05).estimate:.1%} of silent failures at 5% on A (oracle {t3r('A','oracle','capture',0.05).estimate:.1%}), "
    f"while lowering the residual ERROR rate ({t3r('A','M0','resid_err',0.2).estimate:.1%} vs {t3r('A','random','resid_err',0.2).estimate:.1%} random at 20%).")
""")

code(r"""
tiers = rd("stage7/tiers.csv")
tc = tiers[tiers.analysis == "primary"].assign(col=lambda d: d.quantity + " " + d.tier.str.split(" ").str[0])
tc = tc[tc.quantity.isin(["M0", "M1", "oracle"])]
tcp = tc.assign(v=[f"{100 * e:.1f}%" for e in tc.sf_share_captured], B=(100 * tc.budget).round(0).astype(int).astype(str) + "%").pivot_table(index=["heldout", "B"], columns="col", values="v", aggfunc="first").reset_index()
show(tcp, "Accept / review / refer tiers (compact): share of all silent failures falling in the refer tier (top B/2) and the review tier (next B/2)")
show(fmt(tiers, est="sf_share_captured", d=3, keep=["heldout", "analysis", "budget", "quantity", "tier", "workload_share"], name="share of all SF captured [95% CI]"),
     "Accept / review / refer tiers: share of all silent failures in the refer tier (top B/2) and review tier (next B/2) (outputs/stage7/tiers.csv)")
say(f"**What this shows:** at B = 5% the refer tier of M0 on A captures {one(tiers, heldout='A', budget=0.05, quantity='M0', tier='refer (top B/2)').sf_share_captured:.1%} "
    f"of silent failures vs {one(tiers, heldout='A', budget=0.05, quantity='oracle', tier='refer (top B/2)').sf_share_captured:.1%} for the oracle: "
    "the most suspicious cases by uncertainty are not the silent ones. No cost model is used; residual metrics depend only on B.")
ttw = tt[tt.metric == "realised_workload"].copy()
ttw["realised workload [95% CI] (%)"] = [ci(e, l, h, 1, 100) for e, l, h in zip(ttw.estimate, ttw.ci_low, ttw.ci_high)]
ttw["budget"] = (100 * ttw.budget).round(0)
tw = ttw.assign(v=[ci(e, l, h, 1, 100, False) + (" *" if str(f) == "True" else "") for e, l, h, f in zip(ttw.estimate, ttw.ci_low, ttw.ci_high, ttw.workload_flag)], B=ttw.budget.astype(int).astype(str) + "%")
show(tw.pivot_table(index=["heldout", "quantity"], columns="B", values="v", aggfunc="first").reset_index().rename(columns=lambda x: f"nominal {x}" if str(x).endswith("%") else x),
     "Threshold transfer (compact): realised workload on the held-out family, % [95% CI]; * = more than 1 point from the budget")
show(ttw[["heldout", "label", "budget", "quantity", "realised workload [95% CI] (%)", "workload_flag"]],
     "Secondary (Blueprint-literal) threshold transfer: realised workload per held-out family, method and budget (outputs/stage7/threshold_transfer.csv)")
show(fmt(tt[tt.metric != "realised_workload"], d=2, scale=100, keep=["heldout", "label", "budget", "quantity", "metric", "random_resid_sf_same_workload", "workload_difference", "workload_matched", "verdict"]),
     "Threshold transfer: residual rates, capture and contrasts at the realised workloads (remaining rows of outputs/stage7/threshold_transfer.csv)")
say(f"**What this shows:** {n_tt} of {N_tt} cells miss the budget by more than 1 point (A: 0-9%, B: about 0%, C: 37-61% for nominal 5-20%). "
    "Score scales do not transfer between families - the same lesson as the Phase 6 FAR failure - which is why the primary protocol ranks within the "
    "deployment pool. Transferred-threshold residual rates are not comparable across methods.")
bt = rd("stage7/boundary_ties.csv")
btm = bt.groupby(["heldout", "method"]).boundary_tie_share.max().round(4).unstack()
show(btm.reset_index(), "Largest share of the pool in a tie group straddling a budget boundary (max over budgets; outputs/stage7/boundary_ties.csv, full in Appendix A)")
say(f"**What this shows:** for the detectors, ties at the boundary involve at most {btm[['M0','M1','M2']].max().max():.2%} of the pool "
    "(random and oracle are tied by construction and handled by the fractional rule).")
""")

# =====================================================================================================================
md(r"""
<a id="s9"></a>
## 9. Ablations and sensitivity (Phase 8; plan `docs/ablation_plan_frozen.md`, commit f0553df)
Primary endpoint: residual silent-failure rate (`sf_090`) at exact budgets 5 / 10 / 20%; secondary: AUROC / AUPRC / AURC. Declared
primary contrasts (unadjusted): **P1** sign-free M1 - sign-free M0, **P2** sign-free M2 - sign-free M1, **P3** sign-free M1 - non-negative
M1. All other contrasts are SECONDARY with BH-FDR inside each group (`q_bh`). For each group the main text shows the **contrast rows**
(residual rate and AUROC); the complete table (levels and all metrics) is in Appendix A.
""")

code(r"""
def ab_contrasts(g, extra=None):
    d = G[g]; d = d[(d.kind == "contrast") & d.metric.isin(["dresid_sf", "dauroc"])]
    if extra is not None:
        d = d[extra(d)]
    o = fmt(d, d=2, scale=1, keep=["heldout", "label", "c_kind", "level_col", "level", "quantity", "budget", "metric", "primary", "p_boot", "q_bh", "jackknife_min", "jackknife_max", "verdict"])
    m = o.metric == "dresid_sf"
    o.loc[m, "estimate [95% CI]"] = [ci(e, l, h, 2, 100, True) for e, l, h in zip(d.estimate[m.values], d.ci_low[m.values], d.ci_high[m.values])]
    o.loc[~m, "estimate [95% CI]"] = [ci(e, l, h, 3, 1, True) for e, l, h in zip(d.estimate[~m.values], d.ci_low[~m.values], d.ci_high[~m.values])]
    o["budget"] = (100 * o.budget).round(0)
    return o
def vc(g, q):
    d = G[g]; d = d[(d.kind == "contrast") & (d.metric == "dresid_sf") & d.quantity.isin(q)]
    return ", ".join(f"{k}: {v}" for k, v in d.verdict.value_counts().items())
names = {"G1": "signal groups", "G2": "fusion constraint (contains P1-P3)", "G3": "CAM layer / form", "G4": "Mahalanobis vs kNN",
         "G5": "reference size", "G6": "temperature scaling", "G7": "family A severity and corruption", "G8": "family C cue state (v2, v1)", "G9": "label variants"}
""")

code(r"""
# ---- Figure 8a: incremental deltas by ablation (regenerated from the saved tables)
sel = [("G1", "M1-M0"), ("G1", "M2-M1"), ("G2", "sf_M1-sf_M0"), ("G2", "sf_M2-sf_M1"), ("G2", "sf_M1-M1"), ("G4", "M1knn-M1"),
       ("G3", "M1+e_js_norm_l3-M1"), ("G3", "M1+e_js_norm_l2-M1"), ("G3", "sf_M1+e_js_norm_l3-sf_M1"), ("G5", "M1_N500-M1_Nfull"), ("G6", "M0_T1-M0")]
fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
for j, H in enumerate("ABC"):
    for row_, (met, bud, s, ttl) in enumerate((("dresid_sf", 0.2, 100, "delta residual SF rate (pt), 20% budget"), ("dauroc", None, 1, "delta AUROC"))):
        for i, (g, q) in enumerate(sel):
            d = G[g]; d = d[(d.kind == "contrast") & d.level_col.isna() & (d.heldout == H) & (d.quantity == q) & (d.metric == met)]
            if bud is not None:
                d = d[np.isclose(d.budget, bud)]
            if len(d):
                r = d.iloc[0]; prim = q in ("sf_M1-sf_M0", "sf_M2-sf_M1", "sf_M1-M1")
                ax[row_, j].errorbar(s * r.estimate, i, xerr=[[s * (r.estimate - r.ci_low)], [s * (r.ci_high - r.estimate)]], fmt="s" if prim else "o",
                                     color=OI["vermillion"] if prim else OI["blue"], capsize=2)
        ax[row_, j].axvline(0, color="k", lw=0.7); ax[row_, j].set_yticks(range(len(sel))); ax[row_, j].set_yticklabels([q for _, q in sel], fontsize=7)
        ax[row_, j].invert_yaxis(); ax[row_, j].set_title(f"held-out {H}: {ttl}", fontsize=9)
plt.tight_layout(); plt.show()
figcap("8a", "Incremental effect of each ablation (variant - reference), held-out family, 95% paired CIs; red squares = declared primary "
             "contrasts P1-P3, blue = secondary. Negative residual deltas and positive AUROC deltas are improvements.",
       "the only primary contrast that changes the picture is P2 (sign-free CAM drift); allowing a free sign changes nothing for M0 / M1 (P3 = 0).")
""")

code(r"""
for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
    QS = {"G1": ["M1-M0", "M0c-M0", "M2-M1"] + ["s_" + f + "-s_u_log1msp" for f in ("u_logent", "r_maha_pred", "r_knn", "e_js_norm_l4")],
          "G2": ["sf_M0-M0", "sf_M1-M1", "sf_M2-M2", "sf_M1-sf_M0", "sf_M2-sf_M1"],
          "G3": [f"{pre}M1+{c}-{pre}M1" for pre in ("", "sf_") for c in ("e_js_norm_l3", "e_js_norm_l2", "e_js_raw_l4", "e_pca_l4")],
          "G4": ["M1knn-M1", "M2knn-M2", "sf_M1knn-sf_M1", "sf_M2knn-sf_M2"],
          "G5": [f"{pre}M1_N{n}-{pre}M1_Nfull" for pre in ("", "sf_") for n in (500, 1000, 2000)] + ["M1_Nfull-M1"],
          "G6": ["M0_T1-M0", "M1_T1-M1"]}
    show(ab_compact(G[g], QS[g], budgets=(0.1, 0.2) if g == "G2" else (0.2,)),
         f"Ablation {g} - {names[g]} (compact): residual silent-failure rate difference in pt [95% CI] at the 20% budget" + (" and 10%" if g == "G2" else "") + " (v reduces, ^ increases, = no practical difference, ? inconclusive)")
    show(ab_contrasts(g, lambda d: d.level_col.isna()), f"Ablation {g} - {names[g]}: contrast rows (residual SF in pt; AUROC) (outputs/stage8/tables/ablation_{g}.csv)")
    if g == "G1":
        say(f"**What this shows (G1, SECONDARY):** Mahalanobis alone beats log(1-MSP) alone at 20% on A ({rc(g8('A','s_r_maha_pred-s_u_log1msp',0.2,g='G1'),2,100)} pt) and B "
            f"({rc(g8('B','s_r_maha_pred-s_u_log1msp',0.2,g='G1'),2,100)} pt) but not on C ({rc(g8('C','s_r_maha_pred-s_u_log1msp',0.2,g='G1'),2,100)} pt); adding CAM to M0 "
            f"under the non-negative fusion changes nothing on A ({rc(g8('A','M0c-M0',0.2,g='G1'),2,100)} pt). Verdicts: {vc('G1', ['M1-M0','M0c-M0','M2-M1'])}.")
    if g == "G2":
        say(f"**What this shows (G2, contains the PRE-REGISTERED PRIMARY P1-P3):** P3 sf-M1 - M1 = 0.00 pt everywhere (A 20% {rc(g8('A','sf_M1-M1',0.2),2,100)}): "
            f"the optimal uncertainty and Mahalanobis weights are already non-negative, so P1 equals the Phase 7 M1-M0 result "
            f"(A 20% {rc(g8('A','sf_M1-sf_M0',0.2),2,100)}, C 20% {rc(g8('C','sf_M1-sf_M0',0.2),2,100)} pt). **P2 sf-M2 - sf-M1 reduces residual silent failures** on A "
            f"({rc(g8('A','sf_M2-sf_M1',0.05),2,100)} / {rc(g8('A','sf_M2-sf_M1',0.1),2,100)} / {rc(g8('A','sf_M2-sf_M1',0.2),2,100)} pt at 5/10/20%) and C "
            f"({rc(g8('C','sf_M2-sf_M1',0.05),2,100)} / {rc(g8('C','sf_M2-sf_M1',0.1),2,100)} / {rc(g8('C','sf_M2-sf_M1',0.2),2,100)} pt), not on B "
            f"({rc(g8('B','sf_M2-sf_M1',0.2),2,100)} pt at 20%). The fitted CAM weight is negative: confident errors have typical explanations. "
            "Conclusion: a non-negative detector is adequate for uncertainty and representation drift, but not for explanation drift.")
    if g == "G3":
        say(f"**What this shows (G3, SECONDARY):** under non-negative fusion no CAM variant helps; under sign-free fusion layer3 and layer2 CAM also reduce residual "
            f"silent failures on C (layer2 20% {rc(g8('C','sf_M1+e_js_norm_l2-sf_M1',0.2,g='G3'),2,100)} pt) and A at 5-10% (layer3 10% "
            f"{rc(g8('A','sf_M1+e_js_norm_l3-sf_M1',0.1,g='G3'),2,100)} pt). The sign of the weight matters more than the layer or the map distance used.")
    if g == "G4":
        say(f"**What this shows (G4, SECONDARY):** kNN is worse than Mahalanobis on A ({rc(g8('A','M1knn-M1',0.2,g='G4'),2,100)} pt at 20%) and B "
            f"({rc(g8('B','M1knn-M1',0.2,g='G4'),2,100)} pt) and better on C ({rc(g8('C','M1knn-M1',0.2,g='G4'),2,100)} pt): the pre-registered Mahalanobis is "
            "the better choice where representation drift helps at all.")
    if g == "G5":
        say(f"**What this shows (G5, SECONDARY):** a Mahalanobis reference of 500 training images is nearly as good as the full saved bank (A 20% "
            f"{rc(g8('A','M1_N500-M1_Nfull',0.2,g='G5'),2,100)} pt; B 20% {rc(g8('B','M1_N500-M1_Nfull',0.2,g='G5'),2,100)} pt, slightly worse); the full-bank "
            f"refit reproduces the stored Phase 4 reference ({rc(g8('A','M1_Nfull-M1',0.2,g='G5'),2,100)} pt). 'Full' = the saved bank (20,000 / 40,000 images), not the whole training set.")
    if g == "G6":
        say(f"**What this shows (G6, SECONDARY):** temperature scaling barely changes the ranking: before - after = {rc(g8('A','M0_T1-M0',0.2,g='G6'),2,100)} pt (A), "
            f"{rc(g8('B','M0_T1-M0',0.2,g='G6'),2,100)} pt (B), {rc(g8('C','M0_T1-M0',0.2,g='G6'),2,100)} pt (C) at 20% for M0. Calibration matters for the "
            "*label* (which errors count as silent), not for the ranking.")
""")

code(r"""
# ---- G7: family A severity and corruption, Figure 8b
d7 = G["G7"]
def lvl(d, col, v, q, met, b=None):
    r = d[(d.kind == "level") & (d.level_col == col) & (d.level.astype(str) == str(v)) & (d.quantity == q) & (d.metric == met)]
    if b is not None:
        r = r[np.isclose(r.budget, b)]
    return r.iloc[0] if len(r) else None
sev = ["0", "1", "2", "3", "4", "5"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for q in ("random", "M0", "M1", "M2", "sf_M1"):
    for a_, (met, b, s) in zip(ax, (("resid_sf", 0.2, 100), ("auroc", None, 1))):
        if met == "auroc" and q == "random":
            continue
        rr = [lvl(d7, "severity", v, q, met, b) for v in sev]
        y = np.array([r.estimate for r in rr]); lo = np.array([r.ci_low for r in rr]); hi = np.array([r.ci_high for r in rr])
        a_.plot(range(6), s * y, "-o", ms=3, color=COL[q], label=q, ls="--" if q == "M2" else "-")
        a_.fill_between(range(6), s * lo, s * hi, color=COL[q], alpha=0.12, lw=0)
ax[0].set_title("residual SF rate (%) at 20% budget"); ax[1].set_title("silent-failure detection AUROC")
for a_ in ax:
    a_.set_xticks(range(6)); a_.set_xticklabels(["clean"] + sev[1:]); a_.set_xlabel("MedMNIST-C severity")
ax[0].legend(); plt.tight_layout(); plt.show()
m0_5, m1_5, rnd_5 = (lvl(d7, "severity", 5, q, "resid_sf", 0.2) for q in ("M0", "M1", "random"))
figcap("8b", "Family A by severity: residual silent-failure rate at a 20% budget (left) and detection AUROC (right) for M0, M1, M2 (dashed, "
             "identical to M1) and sign-free M1, policy re-applied inside each severity; 95% CIs. SECONDARY.",
       f"M0's AUROC falls from {lvl(d7,'severity',0,'M0','auroc').estimate:.2f} (clean) to {lvl(d7,'severity',5,'M0','auroc').estimate:.2f} (severity 5), "
       f"M1 degrades more slowly ({lvl(d7,'severity',0,'M1','auroc').estimate:.2f} to {lvl(d7,'severity',5,'M1','auroc').estimate:.2f}); at severity 5 both "
       f"leave more residual silent failures than random review ({100*m0_5.estimate:.1f}% / {100*m1_5.estimate:.1f}% vs {100*rnd_5.estimate:.1f}%).")
show(ab_compact(G["G7"], ["M1-M0", "M2-M1", "sf_M1-M1"], level_col="severity", rows_by=["level"], heldouts="A"),
     "Ablation G7 (compact) - family A by severity (0 = clean): residual-rate difference in pt [95% CI] at the 20% budget (v reduces, ^ increases, = none, ? inconclusive)")
show(ab_contrasts("G7"), "Ablation G7 - family A by severity and corruption: contrast rows (outputs/stage8/tables/ablation_G7.csv)")
v7 = d7[(d7.kind == "contrast") & (d7.metric == "dresid_sf") & (d7.quantity == "M1-M0") & np.isclose(d7.budget, 0.2) & (d7.level_col == "condition")]
say(f"**What this shows (G7, SECONDARY):** at 20% M1-M0 reduces residual silent failures for {int((v7.verdict == 'reduces').sum())} of "
    f"{int((v7.level != 'clean').sum())} corruption types (largest: {', '.join(f'{r.level} {100*r.estimate:+.1f} pt' for r in v7.nsmallest(3, 'estimate').itertuples())}); "
    "M2-M1 and sign-free M1 - M1 are 0 at every level.")
""")

code(r"""
# ---- G8: family C cue states, Figure 8c (also used in Section 11)
d8 = G["G8"]
def lv8(ck, st, q, met, b=None):
    r = d8[(d8.kind == "level") & (d8.c_kind == ck) & (d8.level.astype(str) == st) & (d8.quantity == q) & (d8.metric == met)]
    if b is not None:
        r = r[np.isclose(r.budget, b)]
    return r.iloc[0] if len(r) else None
def fig8c():
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for k, (ck, lab) in enumerate((("shortcutv2", "v2 (primary)"), ("shortcut", "v1 (supplement)"))):
        for jj, (met, b, s, t) in enumerate((("auroc", None, 1, "AUROC"), ("resid_sf", 0.2, 100, "residual SF (%) at 20%"))):
            a_ = ax[2 * k + jj]
            for i, q in enumerate(("random", "M0", "M1", "sf_M1", "sf_M2")):
                if met == "auroc" and q == "random":
                    continue
                for xi, st in enumerate(("removed", "reversed")):
                    r = lv8(ck, st, q, met, b)
                    if r is not None:
                        a_.errorbar(xi + (i - 2) * 0.12, s * r.estimate, yerr=[[s * (r.estimate - r.ci_low)], [s * (r.ci_high - r.estimate)]], fmt="o",
                                    color=COL[q], capsize=2, label=q if xi == 0 else None)
            if met == "auroc":
                a_.axhline(0.5, color="k", lw=0.6, ls="--")
            a_.set_xticks([0, 1]); a_.set_xticklabels(["removed", "reversed"]); a_.set_title(f"{lab}: {t}", fontsize=9)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()
fig8c()
figcap("8c", "Family C by cue state (aligned has no silent failures and is omitted): detection AUROC and residual silent-failure rate at 20% "
             "for v2 and v1, 95% CIs. SECONDARY.",
       f"with the strong cue (v2) reversed, every monitor is inverted (M0 AUROC {lv8('shortcutv2','reversed','M0','auroc').estimate:.3f}, M1 "
       f"{lv8('shortcutv2','reversed','M1','auroc').estimate:.3f}); sign-free M2 partly repairs it ({lv8('shortcutv2','reversed','sf_M2','auroc').estimate:.3f}). "
       f"With the weak cue (v1) M0 works well (removed AUROC {lv8('shortcut','removed','M0','auroc').estimate:.3f}) and M1 is worse.")
g8c = ab_compact(G["G8"], ["M1-M0", "M2-M1", "sf_M1-M0", "sf_M2-sf_M1"], level_col="condition", rows_by=["c_kind", "level"])
show(g8c[g8c["level"] != "aligned"].drop(columns=[c for c in g8c.columns if c in ("A", "B")]), "Ablation G8 (compact) - family C by cue state, v2 (shortcutv2) and v1 (shortcut): residual-rate difference in pt [95% CI] at 20%")
show(ab_contrasts("G8"), "Ablation G8 - family C by cue state, v2 and v1: contrast rows (outputs/stage8/tables/ablation_G8.csv)")
say("**What this shows (G8, SECONDARY):** the shortcut hurts every monitor that relies on typicality; the damage is largest when the cue "
    "dominates the decision (v2), and adding representation drift makes it worse in both cue states.")
show(ab_compact(G["G9"], ["M1-M0", "M2-M1", "sf_M1-M0"], rows_by=["label"]), "Ablation G9 (compact) - label variants: residual-rate difference in pt [95% CI] at 20%")
show(ab_contrasts("G9"), "Ablation G9 - label variants sf_090 / sf_095 / sf_tau / sf_099: contrast rows (outputs/stage8/tables/ablation_G9.csv)")
d9 = G["G9"]
def g9(H, lab, q):
    return one(d9[(d9.kind == "contrast")], heldout=H, label=lab, quantity=q, metric="dresid_sf", budget=0.2)
say(f"**What this shows (G9, SECONDARY):** the M1-M0 pattern holds for sf_095 (A {rc(g9('A','sf_095','M1-M0'),2,100)} pt, C {rc(g9('C','sf_095','M1-M0'),2,100)} pt "
    f"at 20%) and shrinks with the degenerate labels (A sf_tau {rc(g9('A','sf_tau','M1-M0'),2,100)} pt, sf_099 {rc(g9('A','sf_099','M1-M0'),2,100)} pt); "
    "M2-M1 is 0 for every label.")
jk = rd("stage8/tables/seed_jackknife_G10.csv")
pj = jk[(jk.metric == "dresid_sf") & jk.quantity.isin(["sf_M1-sf_M0", "sf_M2-sf_M1", "sf_M1-M1"])].copy()
pj["sign flips"] = (pj.jackknife_min * pj.jackknife_max < 0) & (pj.estimate.abs() > 1e-3)
show(pj.round(5), "G10 seed jackknife of the primary contrasts P1-P3 (filtered view of outputs/stage8/tables/seed_jackknife_G10.csv; full in Appendix A)")
say(f"**What this shows:** leaving out any one seed changes the sign of {int(pj['sign flips'].sum())} of {len(pj)} primary contrasts with |d| >= 0.1 pt.")
""")

# =====================================================================================================================
md(r"""
<a id="s10"></a>
## 10. EXPLORATORY, POST-HOC: targeted (confidence-gated) fusion
> **Status: EXPLORATORY-POST-HOC.** Motivated by the Phase 7 result (uncertainty-first ranking spends small budgets on visible errors).
> Reported only here; never a replacement for M0 / M1 / M2 and never in a main table.

**Gated ranking:** cases with calibrated confidence >= 0.90 are ranked first, ordered by a score refitted on the *gated tuning rows*
(G-M0, G-M1: non-negative; G-sf M1, G-sf M2: sign-free); if the gated pool is smaller than the budget, the rest is filled by the plain M0
rank (never needed here). **Limitation:** the gate reuses the 0.90 threshold of the label itself, so it encodes the label definition;
it only uses deployment-available quantities, but its advantage is partly by construction.
""")

code(r"""
gs = rd("stage8/tables/gated_share.csv")
show(gs, "Share of each pool inside the gate (calibrated confidence >= 0.90) (outputs/stage8/tables/gated_share.csv)")
lev = EX[(EX.kind == "level") & EX.metric.isin(["resid_sf", "resid_err"])]
piv = lev.assign(val=[ci(e, l, h, 2, 100, False) for e, l, h in zip(lev.estimate, lev.ci_low, lev.ci_high)], B=(100 * lev.budget).round(0))
piv = piv.pivot_table(index=["heldout", "B", "metric"], columns="quantity", values="val", aggfunc="first").reset_index()
piv = piv[["heldout", "B", "metric", "random", "M0", "M1", "sf_M1", "G_M0", "G_M1", "G_sfM1", "G_sfM2", "oracle"]]
show(piv, "EXPLORATORY, post-hoc - residual silent-failure rate (resid_sf) AND residual overall error rate (resid_err) among accepted cases, % [95% CI]")
say(f"**What this shows:** at 20% the gated M1 ranking leaves {ex('A','G_M1',0.2,'resid_sf').estimate:.2%} silent failures on A "
    f"(plain M1 {ex('A','M1',0.2,'resid_sf').estimate:.2%}, random {ex('A','random',0.2,'resid_sf').estimate:.2%}) and {ex('B','G_M1',0.2,'resid_sf').estimate:.2%} on B "
    f"(plain M1 {ex('B','M1',0.2,'resid_sf').estimate:.2%}). The price is visible errors: its residual ERROR rate on A is {ex('A','G_M1',0.2,'resid_err').estimate:.2%} vs "
    f"{ex('A','M1',0.2,'resid_err').estimate:.2%} for plain M1, because low-confidence errors are no longer referred first.")
exc = EX[(EX.kind == "contrast") & (EX.metric == "dresid_sf")]
show(ab_compact(EX, ["G_M0-M0", "G_M1-M1", "G_sfM1-M1", "G_M0-random", "G_M1-random", "sf_M1-M1"], budgets=(0.05, 0.2)),
     "EXPLORATORY, post-hoc (compact) - targeted variants vs plain M0 / M1 / random: residual-rate difference in pt [95% CI] at 5% and 20% budgets")
show(fmt(exc, d=2, scale=100, signed=True, keep=["table", "heldout", "budget", "quantity", "p_boot", "jackknife_min", "jackknife_max", "verdict"], name="difference in pt [95% CI]"),
     "EXPLORATORY, post-hoc - contrasts of the targeted variants vs plain M0, plain M1 and random (residual SF rate, pt; unadjusted)")
say(f"**What this shows:** gated M1 - plain M1 = A {rc(ex('A','G_M1-M1',0.05),2,100)} / {rc(ex('A','G_M1-M1',0.2),2,100)} pt (5% / 20%), "
    f"B {rc(ex('B','G_M1-M1',0.05),2,100)} / {rc(ex('B','G_M1-M1',0.2),2,100)} pt, C {rc(ex('C','G_M1-M1',0.05),2,100)} / {rc(ex('C','G_M1-M1',0.2),2,100)} pt. "
    f"On C only the gated uncertainty ranking beats random ({rc(ex('C','G_M0-random',0.2),2,100)} pt at 20%); gated M1 does not "
    f"({rc(ex('C','G_M1-random',0.2),2,100)} pt). Ungated sign-free M1 equals M1 ({rc(ex('A','sf_M1-M1',0.2),2,100)} pt).")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a_, H in zip(ax, "ABC"):
    for q in ("random", "oracle", "M0", "M1", "G_M0", "G_M1"):
        g = EX[(EX.kind == "level") & (EX.heldout == H) & (EX.quantity == q) & (EX.metric == "resid_sf")].sort_values("budget")
        a_.errorbar(100 * g.budget, 100 * g.estimate, yerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)], fmt="--" if q == "oracle" else "-o",
                    ms=3, capsize=2, color=COL[q], label=q)
    a_.set_title(f"EXPLORATORY - held-out {H}"); a_.set_xlabel("review budget (%)")
ax[0].set_ylabel("residual SF rate (%)"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.show()
figcap("8d", "EXPLORATORY, post-hoc: residual silent-failure rate of plain vs confidence-gated rankings at 5 / 10 / 20%, with random and oracle references, 95% CIs.",
       "gating closes much of the gap to the oracle on A and B; on the strong shortcut C nothing gets close to the oracle.")
""")

# =====================================================================================================================

md(r"""
<a id="sx"></a>
## 10b. Extensions (post-hoc): deep-ensemble baseline (E1) and in-family vs out-of-family (E2)
> **Status: EXTENSION (post-hoc).** Plan `docs/extension_plan_frozen.md`, commit 6e6f05c, written before any result below. Never a replacement
> for M0 / M1 / M2; no retraining, no GPU, no new image extraction.

### E1. Deep-ensemble uncertainty baseline
The 5 seeds of each model kind were evaluated on IDENTICAL rows in identical order (checked). Per-seed calibrated probability vectors were not
saved, so they are **reconstructed from the saved penultimate features and the frozen final layer** (check: 99.9th percentile of the confidence
error < 1e-3, asserted). For seed unit *s* the ensemble is the **leave-seed-*s*-out ensemble of the other 4 seeds** (`E`), so the seed-level
bootstrap stays valid; the 5-member ensemble including *s* (`E5`) is a sensitivity. The target is unchanged: silent failures (`sf_090`) of the
single seed-*s* classifier. Signals: log(1 - max mean-probability) (`E-msp`), log predictive entropy (`E-ent`), log mutual information (`E-mi`);
`E-fuse` = non-negative fusion of the first two (the ensemble analogue of M0); `E-M1` = `E-fuse` + Mahalanobis. Everything else (rotation, percentile
standardisation, paired cluster bootstrap, exact-budget referral) is as in Sections 6 and 8.
**Limitation:** variability due to which seeds form the ensemble is only partly captured by the seed-level bootstrap.
""")

code(r"""
t10 = rd("stage10/e1_ensemble_baseline.csv"); rd("stage10/ens_reconstruction_checks.csv")
def e10(H, q, m, b=None, kind="level"):
    d = t10[(t10.kind == kind)]
    return one(d, heldout=H, quantity=q, metric=m) if b is None else one(d, heldout=H, quantity=q, metric=m, budget=float(b))
names10 = ["M0", "E_msp", "E_ent", "E_mi", "E_fuse", "M1", "E_M1", "E5_fuse", "E5_M1", "M2"]
lvl = t10[(t10.kind == "level") & t10.metric.isin(["auroc", "auprc", "aurc"])].copy()
lvl["val"] = [ci(e, l, h, 3, 1, False) for e, l, h in zip(lvl.estimate, lvl.ci_low, lvl.ci_high)]
show(lvl.pivot_table(index=["heldout", "metric"], columns="quantity", values="val", aggfunc="first")[names10].reset_index(),
     "E1 - detection of silent failures (sf_090): AUROC, AUPRC, AURC, held-out-family rotation, mean over 5 seed units [95% CI] (outputs/stage10/e1_ensemble_baseline.csv)")
dA = lambda H, m: e10(H, "E_fuse", m).estimate - e10(H, "M0", m).estimate
say(f"**What this shows (EXTENSION):** the ensemble improves on a single model's uncertainty: AUROC E-fuse - M0 = {dA('A','auroc'):+.3f} (A), {dA('B','auroc'):+.3f} (B), "
    f"{dA('C','auroc'):+.3f} (C; point estimates, paired CIs in the contrast table below). On A, M1 ({e10('A','M1','auroc').estimate:.3f}) and E-fuse ({e10('A','E_fuse','auroc').estimate:.3f}) "
    f"are similar, and E-M1 reaches {e10('A','E_M1','auroc').estimate:.3f}; on C the representation signal lowers AUROC even on top of the ensemble "
    f"(E-M1 {e10('C','E_M1','auroc').estimate:.3f} vs E-fuse {e10('C','E_fuse','auroc').estimate:.3f}).")
""")

code(r"""
lv = t10[(t10.kind == "level") & t10.metric.isin(["resid_sf", "resid_err"]) & t10.quantity.isin(names10 + ["random", "oracle"])].copy()
lv["val"] = [ci(e, l, h, 2, 100, False) for e, l, h in zip(lv.estimate, lv.ci_low, lv.ci_high)]; lv["B"] = (100 * lv.budget).round(0)
show(lv.pivot_table(index=["heldout", "B", "metric"], columns="quantity", values="val", aggfunc="first")[["random"] + names10 + ["oracle"]].reset_index(),
     "E1 - exact-budget referral: residual silent-failure rate (resid_sf) and residual overall error rate (resid_err), % [95% CI]")
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a_, H in zip(ax, "ABC"):
    for q in ("random", "oracle", "M0", "M1", "E_fuse", "E_M1"):
        g = t10[(t10.kind == "level") & (t10.heldout == H) & (t10.quantity == q) & (t10.metric == "resid_sf")].sort_values("budget")
        a_.errorbar(100 * g.budget, 100 * g.estimate, yerr=[100 * (g.estimate - g.ci_low), 100 * (g.ci_high - g.estimate)], fmt="--" if q == "oracle" else "-o", ms=3, capsize=2,
                    color={"random": OI["grey"], "oracle": OI["black"], "M0": OI["blue"], "M1": OI["orange"], "E_fuse": OI["green"], "E_M1": OI["vermillion"]}[q], label=q)
    a_.set_title(f"EXTENSION E1 - held-out {H}"); a_.set_xlabel("review budget (%)")
ax[0].set_ylabel("residual silent-failure rate (%)"); ax[0].legend(fontsize=8, ncol=6)
plt.tight_layout(); plt.show()
figcap("10", "EXTENSION E1: residual silent-failure rate vs review budget for single-model M0 / M1, the leave-seed-out ensemble E-fuse and E-M1, with random and oracle references (95% paired CIs).",
       f"on A and B the ensemble alone reproduces most of the 20%-budget gain that M1 had over M0 (A residual: M0 {100*e10('A','M0','resid_sf',0.2).estimate:.1f}%, M1 {100*e10('A','M1','resid_sf',0.2).estimate:.1f}%, "
       f"E-fuse {100*e10('A','E_fuse','resid_sf',0.2).estimate:.1f}%, E-M1 {100*e10('A','E_M1','resid_sf',0.2).estimate:.1f}%); on C no ranking comes close to the oracle and Mahalanobis again hurts.")
cx = t10[(t10.kind == "contrast") & t10.metric.isin(["dresid_sf", "dauroc", "dauprc", "daurc"]) &
         t10.quantity.isin(["E_fuse-M0", "M1-E_fuse", "E_M1-E_fuse", "E_M1-M1", "E_mi-E_msp", "E_ent-E_msp", "E_msp-M0", "E5_fuse-M0", "M1-E5_fuse", "E5_M1-E5_fuse", "M1-M0"])]
o = cx.assign(est=[ci(e, l, h, 2, 100, True) if m == "dresid_sf" else ci(e, l, h, 4 if m == "daurc" else 3, 1, True) for e, l, h, m in zip(cx.estimate, cx.ci_low, cx.ci_high, cx.metric)],
              B=(100 * cx.budget).round(0))
show(o[["heldout", "quantity", "metric", "B", "est", "p_boot", "jackknife_min", "jackknife_max", "verdict"]].rename(columns={"est": "difference [95% CI] (pt for dresid_sf)"}),
     "E1 - paired contrasts (EXTENSION, unadjusted; E-fuse vs M0, M1 vs E-fuse, E-M1 vs E-fuse, variants)")
def cls(v):
    return "better" if v in ("reduces", "improves") else "worse" if v in ("increases", "worse") else "no difference" if v == "no practical difference" else "inconclusive"
rows_ = []
for H in "ABC":
    for m, bs in (("dresid_sf", (0.05, 0.1, 0.2)), ("dauroc", (None,))):
        for b in bs:
            def g(q):
                d = cx[(cx.heldout == H) & (cx.quantity == q) & (cx.metric == m)]
                d = d[np.isclose(d.budget, b)] if b is not None else d
                return d.iloc[0]
            dd, sc_ = (2, 100) if m == "dresid_sf" else (3, 1)
            a_, e_, f_ = g("M1-M0"), g("M1-E_fuse"), g("E_M1-E_fuse")
            rows_.append({"held-out": H, "metric": m, "budget %": (None if b is None else int(100 * b)), "M1 vs M0": f"{rc(a_, dd, sc_)} {a_.verdict}",
                          "M1 vs ensemble E-fuse": f"{rc(e_, dd, sc_)} {e_.verdict}",
                          "verdict class": "same" if cls(a_.verdict) == cls(e_.verdict) else "CHANGES",
                          "Mahalanobis added to the ensemble (E-M1 - E-fuse)": f"{rc(f_, dd, sc_)} {f_.verdict}"})
rr = pd.DataFrame(rows_)
show(rr, "E1 - does the ensemble baseline change a conclusion? (rule fixed in the plan: the verdict class of 'M1 vs ensemble' differs from that of 'M1 vs M0')")
cx20 = lambda H, q: rc(e10(H, q, "dresid_sf", 0.2, "contrast"), 2, 100)
say(f"**What this shows (EXTENSION):** the verdict class changes in {int((rr['verdict class'] == 'CHANGES').sum())} of {len(rr)} (family, metric, budget) cells. "
    f"Residual silent failures at 20%: E-fuse - M0 = {cx20('A','E_fuse-M0')} pt (A), {cx20('B','E_fuse-M0')} pt (B); M1 - E-fuse = {cx20('A','M1-E_fuse')} pt (A) and {cx20('B','M1-E_fuse')} pt (B). "
    "So much of the apparent benefit of representation drift over one model's uncertainty is a benefit of any better uncertainty estimate. Mahalanobis on top of the "
    f"ensemble still helps on A ({cx20('A','E_M1-E_fuse')} pt at 20%) and B ({cx20('B','E_M1-E_fuse')} pt) but hurts on C ({cx20('C','E_M1-E_fuse')} pt). Mutual information is not better "
    f"than the maximum-probability signal (E-mi - E-msp at 20%: A {cx20('A','E_mi-E_msp')}, C {cx20('C','E_mi-E_msp')} pt). The ensemble still leaves "
    f"{100*e10('A','E_fuse','resid_sf',0.05).estimate:.1f}% residual silent failures at 5% on A (random {100*e10('A','random','resid_sf',0.05).estimate:.1f}%): "
    "uncertainty-first ranking, even from an ensemble, does not reach silent failures at small budgets.")
""")

md(r"""
### E2. In-family (source) vs out-of-family detection - saved tables only
Source-distribution rows (family A `clean`, severity 0; family B `id_test`; family C `aligned`) versus shifted rows of the same family, using the
Phase 5 breakdown tables (AUROC / AUPRC with saved CIs) and the Phase 7 breakdowns (residual rate at a 20% budget). The fusion weights are the
held-out-family rotation weights, so "in-family" here means **source-distribution evaluation**, not a fusion tuned inside the same family. The
Phase 5 incremental logistic regression (fitted and evaluated in the same family, in-sample, descriptive) supplies in-family information.
""")

code(r"""
sbk = rd("stage5/secondary_breakdowns.csv"); b7 = rd("stage7/breakdowns.csv")
spec = [("A by severity", "A", ["0", "1", "2", "3", "4", "5"], "A severity (0 = clean = source)"),
        ("B by condition", "B", ["id_test", "ood_c1", "ood_c2"], "B condition (id_test = source)"),
        ("C by condition", "C", ["aligned", "removed", "reversed"], "C cue state (aligned = source)")]
rows = []
for an, H, lvls, lab in spec:
    for l in lvls:
        r = {"family / level": f"{lab}: {l}"}
        for q in ("M0", "M1", "M2"):
            for m in ("AUROC", "AUPRC"):
                d = sbk[(sbk.analysis == an) & (sbk.level.astype(str) == l) & (sbk.quantity == q) & (sbk.metric == m)].iloc[0]
                r[f"{q} {m}"] = ci(d.estimate, d.ci_low, d.ci_high, 3, 1, False) if pd.notna(d.estimate) else "not estimable (no silent failures)"
            d7 = b7[(b7.analysis == an) & (b7.level.astype(str) == l) & (b7.quantity == q) & (b7.metric == "resid_sf") & np.isclose(b7.budget, 0.2)].iloc[0]
            r[f"{q} resid. SF @20% (%)"] = ci(d7.estimate, d7.ci_low, d7.ci_high, 2, 100, False)
        r["SF prevalence"] = round(float(d.sf_rate), 4)
        rows.append(r)
e2 = pd.DataFrame(rows)
show(e2, "E2 - detection (AUROC, AUPRC) and residual silent-failure rate at a 20% budget, source vs shifted rows (outputs/stage5/secondary_breakdowns.csv, outputs/stage7/breakdowns.csv)")
a0, a5 = e2.iloc[0], e2.iloc[5]
say(f"**What this shows (EXTENSION):** M0's detection AUROC is highest on source data and falls with shift on A (clean {a0['M0 AUROC']} -> severity 5 {a5['M0 AUROC']}), "
    f"whereas M1 starts lower on clean data ({a0['M1 AUROC']}) and degrades more slowly ({a5['M1 AUROC']}): representation drift helps where the shift is strong, not in-distribution. "
    "On B the source split has few silent failures and wide CIs; on C the aligned state has none. M2 equals M1 in every row of A and C.")
il = rd("stage5/incremental_logit.csv")
inl = il.groupby("heldout").agg(LR_M1_vs_M0=("LR_M1_vs_M0", "median"), coef_maha=("coef_maha", "median"), LR_M2_vs_M1=("LR_M2_vs_M1", "median"),
                                coef_cam=("coef_cam", "median"), max_p_cam=("wald_p_cam", "max")).round(4).reset_index()
inl["out-of-family M2-M1 dAUROC (monotone fusion)"] = [rc(t2r(H, "M2-M1", "dAUROC"), 4) for H in inl.heldout]
show(inl, "E2 - in-sample (in-family) information: unconstrained logistic regression fitted and evaluated on the held-out family itself, median over seeds, vs the out-of-family monotone result")
pword = 'all Wald p < 1e-30' if inl.max_p_cam.max() < 1e-30 else 'largest Wald p = %.1e' % inl.max_p_cam.max()
say(f"**What this shows (EXTENSION):** inside the family, CAM drift carries significant information given M0 + Mahalanobis (likelihood-ratio statistics "
    f"{inl.LR_M2_vs_M1.min():.0f}-{inl.LR_M2_vs_M1.max():.0f}, {pword}) but with a NEGATIVE coefficient ({inl.coef_cam.min():.2f} to {inl.coef_cam.max():.2f}); "
    "out-of-family, the pre-registered monotone fusion cannot use it (M2-M1 = 0). So the information exists in-sample but does not transfer as a positive-weight alarm; "
    "with a free sign it does help out-of-family (P2, Section 9). **Limitation:** no fusion tuned and tested on disjoint clusters inside the same family was run, so this is "
    "evidence about information content, not a within-family generalisation estimate.")
""")

md(r"""
<a id="s11"></a>
## 11. Shortcut case study (Figure 7)
Does Grad-CAM *see* the shortcut cue, and why do all monitors fail on it? Inputs: the share of CAM mass on the cue patch (Phase 4), the
inversion diagnosis (Phase 5) and the cue-state results (Phase 8). No per-image Grad-CAM example PNGs were saved in the pipeline, so no
qualitative CAM overlays are shown (a limitation).
""")

code(r"""
cm = rd("stage4/cue_mass.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for i, (k, lab) in enumerate((("shortcut", "v1"), ("shortcutv2", "v2"))):
    d = cm[cm.model_kind == k].set_index("condition").loc[["aligned", "removed", "reversed"]]
    ax[0].bar(np.arange(3) + (i - 0.5) * 0.38, d.cue_mass_l3_mean, 0.38, yerr=d.cue_mass_l3_std, color=[OI["sky"], OI["vermillion"]][i], label=f"{lab} layer3", capsize=2)
    ax[0].plot(np.arange(3) + (i - 0.5) * 0.38, d.cue_mass_l4_mean, "k" + ["o", "s"][i], ms=4, label=f"{lab} layer4")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["aligned", "removed", "reversed"]); ax[0].set_title("(a) CAM mass on the cue patch"); ax[0].legend(fontsize=7)
au_ = pd.DataFrame(inv["auroc_sf090"]).T.loc[["pooled", "removed", "reversed"], ["u_log1msp", "r_maha_pred", "r_knn", "e_js_norm_l4"]].astype(float)
for i, c in enumerate(au_.columns):
    ax[1].bar(np.arange(3) + (i - 1.5) * 0.2, au_[c], 0.2, label=c)
ax[1].axhline(0.5, color="k", lw=0.6, ls="--"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(au_.index); ax[1].set_title("(b) single-signal AUROC, v2"); ax[1].legend(fontsize=7)
rp_ = inv["reversed_profile"]
grp = list(rp_)
vals = pd.DataFrame({g: {k.replace("median_pct_", ""): rp_[g][k] for k in rp_[g] if k.startswith("median_pct_")} for g in grp})
for i, g in enumerate(grp):
    ax[2].bar(np.arange(len(vals)) + (i - 1) * 0.27, vals[g], 0.27, label=g)
ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(vals.index, rotation=20, fontsize=7); ax[2].set_ylim(0.5, 1.01)
ax[2].set_title("(c) median val_cal percentile, reversed v2"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
figcap(7, "Shortcut case study: (a) share of Grad-CAM mass on the cue patch per cue state (v1, v2); (b) single-signal detection AUROC pooled and within "
          "each cue state (v2); (c) how atypical correct predictions vs silent failures look under the reversed cue (median val_cal percentile).",
       f"layer3 CAM concentrates on the cue for v2 when it is present (aligned {one(cm, model_kind='shortcutv2', condition='aligned').cue_mass_l3_mean:.3f}, "
       f"reversed {one(cm, model_kind='shortcutv2', condition='reversed').cue_mass_l3_mean:.3f}, removed {one(cm, model_kind='shortcutv2', condition='removed').cue_mass_l3_mean:.3f}) - "
       "the explanation does see the cue (validity check passed) - yet the silently wrong cases are the TYPICAL ones, so typicality monitors invert.")
show(cm, "CAM mass on the cue patch (outputs/stage4/cue_mass.csv)")
say("**What this shows:** Grad-CAM is a valid detector of the cue's *use* (mass moves with the cue), but cue-driven errors are confident and "
    "typical; post-hoc monitoring cannot flag them unless the monitor is allowed to treat typicality as suspicious (sign-free CAM, Section 9).")
fig8c()
figcap("7b", "Cue-state results repeated from Figure 8c for the case study.", "the inversion persists for every detector; sign-free M2 recovers part of it.")
dr = rd("stage4/dose_response.csv")
show(dr.round(4), "Dose response of every signal over severity per corruption (outputs/stage4/dose_response.csv)")
say(f"**What this shows:** {int(dr.monotone_nondecreasing.astype(str).eq('True').sum())} of {len(dr)} (corruption, signal) curves are monotone "
    "non-decreasing in severity; the signals respond to shift magnitude as intended.")
png("stage4/figures/dose_response_A.png", width=850)
figcap("S5", "Saved Phase 4 dose-response figure for family A.", "signals rise with severity for most corruptions.")
png("stage3/figures/cue_palette_v2.png", width=500)
figcap("S6", "The v2 shortcut cue palette (one colour per class).", "the cue is a small corner patch, invisible to a pathologist's diagnosis.")
""")

# =====================================================================================================================
md(r"""
<a id="s12"></a>
## 12. Blueprint compliance checklist and deviations
Status: **PASS** (done as specified), **DEVIATION** (done differently, failed a pre-registered check, or not done), **N-A** (not
applicable). Every evidence path is checked for existence when the notebook runs.
""")

code(r"""
C = [
 ("0 Reproducibility", "single YAML config, no hard-coded paths", "PASS", "configs/full.yaml"),
 ("0 Reproducibility", "environment record (versions, GPU, seed)", "PASS", "outputs/env_record.json"),
 ("0 Reproducibility", "resumable stage scripts + smoke config", "PASS", "configs/smoke.yaml"),
 ("0 Reproducibility", "short + sustained GPU benchmark", "PASS", "outputs/bench_sustained.json"),
 ("1 Data", "PathMNIST 128 px, Camelyon17-WILDS v1.0 96 px, MedMNIST-C v0.1.0", "PASS", "outputs/stage1/table1.csv"),
 ("1 Data", "Camelyon17 from the official source", "DEVIATION", "outputs/stage1/table1.csv"),
 ("1 Data", "frozen splits with uid and group / patient", "PASS", "outputs/stage1/splits.parquet"),
 ("1 Data", "exact + near-duplicate audit", "PASS", "outputs/stage1/dedup_report.csv"),
 ("1 Data", "automated no-overlap assertions (patient, slide, hospital)", "PASS", "outputs/stage1/leakage_audit.json"),
 ("1 Data", "file checksums", "PASS", "outputs/stage1/checksums.sha256"),
 ("1 Data", "sub-dataset manifest (group-aware, fixed seed)", "PASS", "outputs/stage1/subset_manifest.parquet"),
 ("2 Classifier", "ResNet-18 (ImageNet init, documented), 5 seeds", "PASS", "outputs/stage2/metrics_per_seed_pathmnist.csv"),
 ("2 Classifier", "early stopping on source val; temperature scaling on val_cal", "PASS", "outputs/stage2/calibration_camelyon17.csv"),
 ("2 Classifier", "ECE / NLL / Brier before and after; reliability diagrams", "PASS", "outputs/stage2/figures/reliability_pathmnist.png"),
 ("2 Classifier", "second architecture supplement", "DEVIATION", "README.md"),
 ("3 Shifts", "A MedMNIST-C severities 1-5 (11 corruptions)", "PASS", "outputs/stage3/per_corruption_accuracy.csv"),
 ("3 Shifts", "B held-out hospitals", "PASS", "outputs/stage3/family_summary.csv"),
 ("3 Shifts", "C synthetic shortcut, 5 seeds, present / removed / reversed", "PASS", "outputs/stage3/shortcut_states.csv"),
 ("3 Shifts", "X CRC-VAL-HE-7K supplement", "PASS", "outputs/stage3/family_summary.csv"),
 ("3 Shifts", "monotone accuracy drop with severity; shortcut reliance measured", "PASS", "outputs/stage3/sanity_stage3.json"),
 ("4 Signals", "MSP, entropy, energy", "PASS", "outputs/stage4/per_signal_auroc.csv"),
 ("4 Signals", "deep-ensemble uncertainty baseline (leave-seed-out ensemble, EXTENSION Section 10b)", "PASS", "outputs/stage10/e1_ensemble_baseline.csv"),
 ("4 Signals", "MC-dropout uncertainty", "DEVIATION", "README.md"),
 ("4 Signals", "class-conditional Mahalanobis, kNN, batch MMD", "PASS", "outputs/stage6/power_pooled.csv"),
 ("4 Signals", "Grad-CAM layer3 + layer2 / layer4 sensitivity", "PASS", "outputs/stage4/refs_l2_checksums.csv"),
 ("4 Signals", "Grad-CAM++ / ScoreCAM / Integrated Gradients alternatives", "DEVIATION", "README.md"),
 ("4 Signals", "normalisation by benign-transform variability", "PASS", "outputs/stage4/transform_variability.csv"),
 ("4 Signals", "validity check of the explanation on family C (cue mass, randomisation)", "PASS", "outputs/stage4/cue_mass.csv"),
 ("4 Signals", "references from source train / val_cal only (asserted, checksummed)", "PASS", "outputs/stage4/refs_checksums.csv"),
 ("5 Sample level", "silent-failure label by the 95%-precision rule", "DEVIATION", "outputs/stage4/frozen_tau.csv"),
 ("5 Sample level", "nested M0 < M1 < M2 with monotone fusion", "PASS", "outputs/stage5/fusion_weights.csv"),
 ("5 Sample level", "held-out-family rotation A / B / C", "PASS", "outputs/stage5/table2.csv"),
 ("5 Sample level", "AUROC / AUPRC / AURC, two-level bootstrap >= 2,000, paired deltas", "PASS", "outputs/stage5/table2.csv"),
 ("5 Sample level", "verdict generated from the numbers; jackknife", "PASS", "outputs/stage5/summary_stage5.md"),
 ("5 Sample level", "in-family vs out-of-family comparison (EXTENSION Section 10b: source-vs-shifted + in-sample information only; no within-family disjoint-cluster tuning, so it stays DEVIATION)", "DEVIATION", "outputs/stage5/secondary_breakdowns.csv"),
 ("6 Batch", "windows of 100 with shifted fractions 0.1-1", "PASS", "outputs/stage6/power_cells.csv"),
 ("6 Batch", "5% FAR thresholds from the source null", "PASS", "outputs/stage6/thresholds_silent_bounds.csv"),
 ("6 Batch", "actual FAR within 3-7% on held-out source windows", "DEVIATION", "outputs/stage6/far_table.csv"),
 ("6 Batch", "power and detection delay (Figure 5)", "PASS", "outputs/stage6/a1/detection_delay.csv"),
 ("7 Referral", "budgets 5 / 10 / 20%; residual SF rate vs random, M0, M1, M2", "PASS", "outputs/stage7/table3_referral.csv"),
 ("7 Referral", "thresholds tuned on two families give a realised workload near the budget", "DEVIATION", "outputs/stage7/threshold_transfer.csv"),
 ("8 Ablation", "severity, corruption type, cue present / removed / reversed", "PASS", "outputs/stage8/tables/ablation_G7.csv"),
 ("8 Ablation", "raw vs temperature-scaled", "PASS", "outputs/stage8/tables/ablation_G6.csv"),
 ("8 Ablation", "Grad-CAM sensitivity (layer, PCA, resolution)", "PASS", "outputs/stage8/tables/ablation_G3.csv"),
 ("8 Ablation", "removal of artefact (non-tissue) patches in Camelyon", "DEVIATION", "outputs/stage1/subset_manifest.parquet"),
 ("8 Ablation", "seed stability; BH-FDR in secondary families", "PASS", "outputs/stage8/tables/seed_jackknife_G10.csv"),
 ("9 Synthesis", "Figure 1 study schematic; evidence map; automated conclusions", "PASS", "silent_failure_report.ipynb"),
 ("9 Synthesis", "one self-contained final notebook + README + requirements", "PASS", "requirements.txt"),
]
comp = pd.DataFrame(C, columns=["stage", "Blueprint item", "status", "evidence"])
comp["evidence exists"] = [(ROOT / p).exists() for p in comp.evidence]
assert comp["evidence exists"].all(), comp[~comp["evidence exists"]]
show(comp, "Blueprint compliance checklist", keep=True)
cnt = comp.status.value_counts()
say(f"**What this shows:** {cnt.get('PASS', 0)} PASS, {cnt.get('DEVIATION', 0)} DEVIATION, {cnt.get('N-A', 0)} N-A; every evidence file exists. "
    "Not-done items (second architecture, MC-dropout, alternative attribution methods, within-family in-family analysis, artefact-patch ablation) were "
    "dropped for time and are listed as deviations rather than hidden; the ensemble baseline was added afterwards as a post-hoc extension.")
""")

md(r"""
### Deviations from the plan (all disclosed in the frozen plans or the README)
| # | deviation | why | consequence |
|---|---|---|---|
| 1 | Primary label `sf_090` instead of the Blueprint `sf_tau` | 19/20 classifiers already exceed 95% precision on val_cal, so `sf_tau` labels nearly every error (Section 1) | fixed before any Phase 5 result; `sf_tau`, `sf_095`, `sf_099` reported as sensitivities |
| 2 | Camelyon17-WILDS read from the unofficial HF mirror `wltjr1007/Camelyon17-WILDS` | official download not feasible on this machine | counts verified against WILDS v1.0 metadata; TLS never disabled |
| 3 | Phase 6 amendment A1 (FAR-matched thresholds) | pre-registered FAR check failed (val_cal vs id_test not exchangeable at window level) | committed (4ea78c3) before any A1 number; pre-registered results kept and shown side by side |
| 4 | Relaxed logit-reconstruction check in Phase 8 | fp16 feature storage amplified by 1/T for saturated v2 models (max error 1.4e-3) | check = 99.9th percentile < 1e-3 and max < 1e-2; see `outputs/stage8/augment_checks.csv` |
| 5 | Family C strength chosen by a pre-registered pilot on `val_es` | the weak v1 shortcut produced too few silent failures | v2 (100% cue, 16 px) primary; v1 supplement; strength is a tuned dial (limitation) |
| 6 | Reference bank sizes | Mahalanobis / kNN references use 20,000 PathMNIST / 40,000 Camelyon training images; Phase 8 "full" = this bank | not the full training set |
| 7 | Phase 6 plan wording | log-scale mean uncertainty is not blind to silent windows | wording corrected in README; definitions unchanged |
| 8 | X not re-analysed in Phase 8 | supplement already covered in Phases 5 and 7 | none for primary results |
""")

code(r"""
aug = rd("stage8/augment_checks.csv")
show(aug, "Phase 8 augmentation checks: Mahalanobis refit vs stored reference and logit reconstruction (outputs/stage8/augment_checks.csv)")
say(f"**What this shows:** the full-bank refit matches the stored reference (minimum rank correlation {aug.spearman_full_refit_vs_stored.min():.6f}); "
    f"the largest confidence reconstruction error is {aug.logit_conf_max_abs_diff.max():.4f} and the largest 99.9th percentile {aug.logit_conf_p999_abs_diff.max():.5f}.")
""")

# =====================================================================================================================
md(r"""
<a id="s13"></a>
## 13. Honest limitations
1. **Few Camelyon patients.** Source splits contain 10 / 4 / 6 / 6 patients (train / val_es / val_cal / id_test); the patient-level
   bootstrap therefore resamples few units, and Camelyon CIs are wide (e.g. Table 3 random residual on B spans about 8-17%).
2. **Tumour-prior gap.** Camelyon train is balanced (50% tumour) while val_cal / id_test / held-out hospitals are about 42%; temperature
   and thresholds are fitted on the 42% splits.
3. **Uneven per-hospital cells.** Some split x hospital cells hold a single patient (Section 2), so per-hospital conclusions are fragile.
4. **No patient IDs in PathMNIST.** Near-duplicate clusters approximate the independent unit; images from one patient may still fall
   into different clusters and splits.
5. **Shortcut strength is a tuned dial.** Family C v2 was chosen to produce strong failures; conclusions about shortcuts depend on that
   strength (v1 behaves very differently, Section 9 G8).
6. **Absolute residual rates are not comparable across families** (silent-failure prevalence ranges from about 1% to 43% of the pool);
   compare methods within a family only.
7. **Single backbone** (ResNet-18) and a single explanation method (Grad-CAM); the second architecture and alternative attributions were
   not run.
8. **Mechanism of the negative CAM weight is a hypothesis.** "Confident errors have typical explanations" is consistent with the
   incremental logistic regression, the inversion diagnosis and P2, but was not tested causally.
9. **Batch thresholds do not transfer** between source splits (Phase 6) and score thresholds do not transfer between families (Phase 7):
   any deployment would need local calibration.
10. **Ensemble baseline (extension).** The ensemble is built from the same 5 training runs; its probabilities are reconstructed from fp16 features; the seed
    level partly ignores which members form the ensemble (Section 10b).
11. **No qualitative CAM overlays** were saved; the case study is quantitative only.
""")

# =====================================================================================================================
md(r"""
<a id="s14"></a>
## 14. Reproducibility
""")

code(r"""
tags = pd.DataFrame([("phase-0", "aaf37ad", "repo, environment, configs, benchmarks, tests, smoke"),
                     ("phase-1", "3c53980", "data, frozen splits, leakage audit (+ Phase 1b patient assignment)"),
                     ("phase-2", "dc51f87", "source classifiers, temperature scaling"),
                     ("phase-3", "96c4597", "shift families A, B, C (v1, v2), X"),
                     ("phase-4", "8a6bba1", "monitoring references and signals"),
                     ("phase-5", "4cdc828", "sample-level detection (plan e388dd6)"),
                     ("phase-6", "ef592c7", "batch monitoring (plan 931fb36, amendment 4ea78c3)"),
                     ("phase-7", "d51da65", "referral policy (plan aa962a9)"),
                     ("phase-8", "83d81db", "ablations (plan f0553df)"),
                     ("phase-9", "0c9cdd8", "final notebook (phase-9b: silent-share fix, known gaps, 73ded51)"),
                     ("phase-10", "(this release)", "extensions E1 ensemble baseline, E2 in-family vs out-of-family (plan 6e6f05c)")], columns=["tag", "commit", "content"])
try:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=10).stdout.strip()
    have = set(subprocess.run(["git", "tag"], capture_output=True, text=True, cwd=ROOT, timeout=10).stdout.split())
    tags["tag present in this clone"] = [t in have for t in tags.tag]
except Exception:
    head = "git not available"
show(tags, f"Phase tags (notebook executed at commit {head})")
env = rd("env_record.json")
say(f"**Environment of the experiments:** Python {env['python']}, PyTorch {env['torch']} (CUDA {env['torch_cuda']}), "
    f"GPU {env['gpu']['name']} ({env['gpu']['vram_gb']} GB), {env['ram_gb']['total']} GB RAM, {env['platform']}. Seeds: classifiers 0-4; "
    "bootstrap seed 0; all window / split RNGs seeded (see `configs/full.yaml`).")
cal_ = pd.concat([rd("stage2/calibration_pathmnist.csv"), rd("stage2/calibration_camelyon17.csv")])
rt = pd.DataFrame([
    ("2 classifiers, PathMNIST (5 seeds, GPU)", round(cal_[cal_.dataset == "pathmnist"].train_seconds.sum() / 60, 1), "outputs/stage2/calibration_pathmnist.csv"),
    ("2 classifiers, Camelyon (5 seeds, GPU)", round(cal_[cal_.dataset == "camelyon17"].train_seconds.sum() / 60, 1), "outputs/stage2/calibration_camelyon17.csv"),
    ("3 shortcut training (per seed, GPU)", "22-28", "README.md (Stage 3)"),
    ("3 corruption generation (4 workers)", "~25", "README.md (Stage 3)"),
    ("5 sample-level analysis (CPU)", 53, "README.md (Stage 5)"),
    ("6 extraction (GPU) + windows + analysis, 2 variants", "8 + 18 + 6", "README.md (Stage 6)"),
    ("7 referral analysis (CPU)", 26, "README.md (Stage 7)"),
    ("8 ablations (CPU)", 143, "README.md (Stage 8)"),
], columns=["phase / step", "minutes", "source"])
show(rt, "Measured runtimes")
say("**How to regenerate everything** (needs the raw data - tens of GB of downloads - and a CUDA GPU; see README): "
    "`python scripts/run_all.py --config configs/full.yaml` (data, frozen splits, classifiers), "
    "`python scripts/run_stage3.py --steps sample,shortcut,corrupt,inferA,tables` (shift families), `python scripts/run_stage4.py` "
    "(references and signals), then `python scripts/run_stage5.py`, `run_stage6.py --steps extract,windows,analysis,windows_a1,analysis_a1`, `run_stage7.py`, `run_stage8.py` "
    "(each with `--config configs/full.yaml`), then `python scripts/build_report_notebook.py` and "
    "`jupyter nbconvert --to notebook --execute --inplace silent_failure_report.ipynb`. A pipeline smoke test runs in minutes with "
    "`python scripts/run_all.py --config configs/smoke.yaml`.")
""")

# =====================================================================================================================
md(r"""
<a id="sA"></a>
## Appendix A: all remaining tables
Every saved result table not shown in full above, in full, formatted with 95% CIs where the table has them.
""")

code(r"""
IN_APPENDIX = True
for i, (t_, d_) in enumerate(DEFERRED, 1):
    show(d_, f"A.D{i} {t_}")
def full(rel, title, d=4):
    t = rd(rel)
    if {"estimate", "ci_low", "ci_high"} <= set(t.columns):
        t = fmt(t, d=d)
    show(t, f"{title} ({'outputs/' + rel})")

full("stage5/sensitivity.csv", "A.1 Phase 5 sensitivity analyses (all rows)")
full("stage5/secondary_breakdowns.csv", "A.2 Phase 5 secondary breakdowns (all rows)")
show(fw.round(4), "A.3 Phase 5 fusion weights per seed (outputs/stage5/fusion_weights.csv)")
full("stage5/tau99.csv", "A.4 Phase 5 tau for the 99% precision label")
show(pd.DataFrame(rd("stage5/tie_report.json")).T.reset_index().rename(columns={"index": "model kind"}), "A.5 Phase 5 tie report (share of tied scores per model kind)")
full("stage4/per_signal_auroc.csv", "A.6 Phase 4 descriptive per-signal AUROC")
full("stage4/transform_variability.csv", "A.7 Phase 4 benign-transform CAM variability")
""")

code(r"""
for v, base in (("pre-registered", "stage6/"), ("A1", "stage6/a1/")):
    show(rd(base + "far_table.csv"), f"A.8 Phase 6 FAR table, {v} ({base}far_table.csv)")
    show(fmt(rd(base + "power_pooled.csv"), est="power", d=3, keep=["H", "kind", "size", "mask", "stat", "n_windows"]), f"A.9 Phase 6 pooled power, {v}")
    show(fmt(rd(base + "contrasts.csv"), est="diff", d=3, signed=True), f"A.10 Phase 6 all contrasts (primary, CAM layer, variants), {v}")
    show(rd(base + "power_cells.csv").round(3), f"A.11 Phase 6 power per cell and fraction, {v}")
    show(rd(base + "detection_delay.csv"), f"A.12 Phase 6 detection delay, all onsets, {v}")
    show(rd(base + "seed_spread.csv").round(3), f"A.13 Phase 6 seed spread of power on silent windows, {v}")
    show(rd(base + "thresholds_silent_bounds.csv").round(4), f"A.14 Phase 6 thresholds and silent-window bounds, {v}")
    show(rd(base + "combined_weights.csv").round(4), f"A.15 Phase 6 combined-monitor weights, {v}")
""")

code(r"""
full("stage7/fig6_data.csv", "A.16 Phase 7 Figure 6 data (budget grid)")
full("stage7/sensitivity.csv", "A.17 Phase 7 sensitivity analyses")
full("stage7/breakdowns.csv", "A.18 Phase 7 per-corruption / severity / condition breakdowns")
full("stage7/seed_jackknife.csv", "A.19 Phase 7 seed jackknife")
show(bt, "A.20 Phase 7 boundary ties (outputs/stage7/boundary_ties.csv)")
show(rd("stage7/fusion_weights.csv").round(4), "A.21 Phase 7 fusion weights (asserted equal to Phase 5)")
""")

code(r"""
for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"):
    show(fmt(G[g], d=4, keep=["heldout", "label", "c_kind", "level_col", "level", "kind", "group", "quantity", "budget", "metric", "n_per_seed", "prevalence", "primary", "p_boot", "q_bh", "jackknife_min", "jackknife_max", "verdict"]),
         f"A.22 Ablation {g} - {names[g]}, full table (outputs/stage8/tables/ablation_{g}.csv)")
show(fmt(EX, d=4), "A.23 EXPLORATORY, post-hoc targeted-fusion table, full (outputs/stage8/tables/exploratory_targeted.csv)")
show(jk, "A.24 G10 seed jackknife, all contrasts (outputs/stage8/tables/seed_jackknife_G10.csv)")
show(fmt(t10_, d=4, keep=["table", "heldout", "label", "kind", "group", "quantity", "budget", "metric", "n_per_seed", "prevalence", "p_boot", "jackknife_min", "jackknife_max", "verdict"]),
     "A.31 EXTENSION E1 full table (outputs/stage10/e1_ensemble_baseline.csv)")
""")

code(r"""
show(rd("stage3/metrics_per_seed.csv").round(4), "A.25 Phase 3 behaviour per condition and seed")
show(rd("stage4/per_signal_auroc_per_seed.csv").round(4), "A.26 Phase 4 per-signal AUROC per seed")
show(rd("stage1/camelyon_spatial_overlap.csv"), "A.27 Camelyon spatial overlap of patches per slide (full download)")
show(rd("stage1/camelyon_spatial_overlap_subset.csv"), "A.28 Camelyon spatial overlap of patches per slide (used subset)")
show(rd("stage1b/candidates.csv").drop(columns=["assignment"]), "A.29 Phase 1b candidate patient assignments")
show(pd.concat([rd("stage2/seed_stability_pathmnist.csv").assign(dataset="pathmnist"), rd("stage2/seed_stability_camelyon17.csv").assign(dataset="camelyon17")]),
     "A.30 Phase 2 seed stability")
""")

md(r"""
<a id="sB"></a>
## Appendix B: file index
Every file this notebook read (generated automatically), with a one-line description.
""")

code(r"""
DESC = {
 "table1": "datasets, splits, sizes, licences", "splits.parquet": "frozen split table (uid, split, group, patient, hospital, label)",
 "dedup_report": "near-duplicate audit", "leakage_audit": "automated leakage checks", "near_duplicate_pairs": "example near-duplicate pairs",
 "best_candidate_detail": "selected Camelyon patient assignment", "candidates": "candidate patient assignments",
 "camelyon_spatial_overlap": "patch overlap per slide", "metrics_per_seed": "per-seed metrics", "calibration_": "temperatures and training length",
 "reliability": "reliability diagrams", "seed_stability": "seed stability", "condition_summary": "behaviour per condition",
 "family_summary": "behaviour per family", "per_corruption": "per-corruption x severity behaviour", "shortcut_states": "cue-state behaviour",
 "examples.png": "example shifted images", "cue_palette": "shortcut cue colours", "frozen_tau": "Blueprint threshold tau per model",
 "cue_mass": "CAM mass on the cue", "dose_response": "signal dose response", "per_signal_auroc": "descriptive per-signal AUROC",
 "transform_variability": "benign-transform CAM variability", "fig3_data": "single-signal AUROC (Fig 3)", "table2": "Table 2 detection",
 "fusion_weights": "fusion weights", "incremental_logit": "incremental logistic regression", "sensitivity": "sensitivity analyses",
 "secondary_breakdowns": "secondary breakdowns", "tau99": "99% precision tau", "tie_report": "score ties", "inversion_diagnosis": "family C inversion diagnosis",
 "far_table": "actual false-alarm rates", "power_pooled": "pooled batch power", "contrasts": "batch contrasts", "power_cells": "batch power per cell",
 "detection_delay": "detection delay", "thresholds_silent_bounds": "batch thresholds", "combined_weights": "combined-monitor weights",
 "fig6_data": "referral curves (Fig 6)", "table3_referral": "Table 3 referral", "tiers": "refer / review tiers", "threshold_transfer": "threshold-transfer check",
 "boundary_ties": "budget boundary ties", "breakdowns": "referral breakdowns", "seed_jackknife": "seed jackknife", "ablation_G": "ablation group table",
 "exploratory_targeted": "EXPLORATORY targeted fusion", "gated_share": "gated share of pools", "augment_checks": "Phase 8 refit / reconstruction checks",
 "env_record": "environment record", "e1_ensemble_baseline": "EXTENSION E1 ensemble baseline", "ens_reconstruction": "ensemble probability reconstruction checks", "fig10_e1": "EXTENSION E1 referral figure", "candidates.csv": "candidate assignments",
}
def describe(p):
    for k, v in DESC.items():
        if k in p:
            return v
    return "result file"
idx = pd.DataFrame({"file": sorted(set(READ))})
idx["description"] = idx.file.map(describe)
idx["size (KB)"] = [round((ROOT / f).stat().st_size / 1024, 1) for f in idx.file]
show(idx, "Files read by this notebook")
print(f"notebook executed in {time.time() - T_START:.0f} s")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, ROOT / "silent_failure_report.ipynb")
print("wrote", ROOT / "silent_failure_report.ipynb", len(cells), "cells")
