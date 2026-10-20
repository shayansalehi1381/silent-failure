"""Leakage audit, Table 1, checksums, manifest hashing and the near-duplicate pair figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDED = "excluded"


# ------------------------------------------------------------------ integrity assertions ----
def assert_unique_uid(df: pd.DataFrame) -> None:
    dup = df["uid"][df["uid"].duplicated()]
    assert dup.empty, f"{len(dup)} duplicated uid, e.g. {dup.iloc[:3].tolist()}"


def assert_group_disjoint(df: pd.DataFrame, col: str, split_col: str = "split") -> None:
    """No value of `col` (cluster id, slide, ...) may occur in more than one split (excluded rows ignored)."""
    d = df[df[split_col] != EXCLUDED]
    n = d.groupby(col)[split_col].nunique()
    bad = n[n > 1]
    assert bad.empty, f"{len(bad)} values of '{col}' cross splits, e.g. {bad.index[:3].tolist()}"


def assert_splits_nonempty_all_classes(df: pd.DataFrame, n_classes: int, splits: list[str]) -> None:
    for s in splits:
        lab = df.loc[df["split"] == s, "label"]
        assert len(lab) > 0, f"split '{s}' is empty"
        assert lab.nunique() == n_classes, f"split '{s}' has {lab.nunique()} of {n_classes} classes"


def assert_fractions(df: pd.DataFrame, target: dict[str, float], tol: float) -> dict:
    n = df["split"].isin(target).sum()
    got = {s: float((df["split"] == s).sum() / n) for s in target}
    for s, f in target.items():
        assert abs(got[s] - f) <= tol, f"split '{s}' fraction {got[s]:.4f} deviates from {f} by > {tol}"
    return got


def assert_no_pair_across(i: np.ndarray, j: np.ndarray, rms: np.ndarray, thr: float,
                          split_i: np.ndarray, split_j: np.ndarray, ignore=(EXCLUDED,)) -> int:
    """Number of near-duplicate pairs (rms <= thr) whose two members are in different, non-ignored splits."""
    k = rms <= thr
    a, b = split_i[i[k]], split_j[j[k]]
    keep = ~(np.isin(a, ignore) | np.isin(b, ignore)) & (a != b)
    return int(keep.sum())


# ------------------------------------------------------------------ hashing ---------------
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def manifest_hash(df: pd.DataFrame, cols: list[str]) -> str:
    """Order-independent hash of selected manifest columns (canonical CSV bytes)."""
    d = df[cols].sort_values(cols[0], kind="stable").reset_index(drop=True)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def write_checksums(files: list[Path], out: Path, root: Path) -> dict[str, str]:
    sums = {}
    for f in files:
        if f.exists():
            try:
                name = str(f.relative_to(root))
            except ValueError:               # outside the repo (e.g. a temp output dir in tests)
                name = str(f)
            sums[name.replace("\\", "/")] = sha256_file(f)
    out.write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())), encoding="utf-8")
    return sums


# ------------------------------------------------------------------ Table 1 ---------------
def table1(df: pd.DataFrame, info: dict[str, dict]) -> pd.DataFrame:
    """One row per (dataset, split). `info[dataset]` carries domain/family/licence text and class names."""
    rows = []
    for (ds, split), g in df.groupby(["dataset", "split"], sort=False):
        meta = info[ds]
        cn = meta["class_names"]
        cls = g["label"].value_counts().sort_index()
        rows.append({
            "dataset": ds, "split": split, "domain": meta["domains"].get(split, meta["domains"]["default"]),
            "family": meta["families"].get(split, meta["families"]["default"]),
            "role": meta["roles"].get(split, ""), "n": len(g), "n_groups": int(g["group"].nunique()),
            "class_counts": json.dumps({cn[int(k)]: int(v) for k, v in cls.items()}),
            "hospital_counts": json.dumps({str(int(k)): int(v) for k, v in g["hospital"].value_counts().sort_index().items()
                                           if k >= 0}),
            "flagged_nontissue": int(g["flag_nontissue"].sum()) if "flag_nontissue" in g else 0,
            "access_licence": meta["access"]})
    return pd.DataFrame(rows)


def to_markdown(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in t.itertuples(index=False):
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figure ----------------
def pair_grid(pairs: list[tuple[np.ndarray, np.ndarray, str]], out: Path, ncols: int = 4) -> None:
    """pairs: (image_a, image_b, caption). Draws a grid, each cell shows a|b side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = max(len(pairs), 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.0 * nrows + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (a, b, cap) in zip(axes.ravel(), pairs):
        ax.imshow(np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], axis=1))
        ax.set_title(cap, fontsize=7)
    if not pairs:
        axes[0, 0].text(0.5, 0.5, "no near-duplicate pairs found", ha="center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
