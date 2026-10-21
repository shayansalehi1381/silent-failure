"""Image data for the source classifiers: per-split raw uint8 .npy caches built from the frozen splits.

The row order of every split is the row order of `splits.parquet` (filtered by dataset and split), so
saved predictions can be matched to the frozen manifest by position and by uid. Caches live in the
git-ignored data directory; they are built once (streaming the compressed sources, labels verified).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from sfd.data.pathmnist import iter_npz_chunks

SPLITS = {"pathmnist": ["train", "val_es", "val_cal", "id_test", "nct7k_test"],
          "camelyon17": ["train", "val_es", "val_cal", "id_test", "ood_c1", "ood_c2"]}
N_CLASSES = {"pathmnist": 9, "camelyon17": 2}


@dataclass
class SplitData:
    name: str
    images: object            # uint8 N x H x W x 3: np.memmap / ndarray, or a torch tensor on the GPU
    labels: np.ndarray        # int64
    uid: np.ndarray           # str
    patient: np.ndarray       # int64 (-1 for PathMNIST)


def split_frame(df: pd.DataFrame, dataset: str, split: str, max_rows: int | None = None) -> pd.DataFrame:
    d = df[(df["dataset"] == dataset) & (df["split"] == split)].reset_index(drop=True)
    return d.iloc[:max_rows] if max_rows else d


def _paths(cache_dir: Path, dataset: str, split: str) -> tuple[Path, Path]:
    return cache_dir / f"{dataset}_{split}.npy", cache_dir / f"{dataset}_{split}.ok.json"


def _uid_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(frame["uid"]).encode()).hexdigest()[:16]


def _is_built(cache_dir: Path, dataset: str, split: str, frame: pd.DataFrame) -> bool:
    npy, ok = _paths(cache_dir, dataset, split)
    if not (npy.exists() and ok.exists()):
        return False
    meta = json.loads(ok.read_text())
    return meta["n"] == len(frame) and meta["uid_hash"] == _uid_hash(frame)


def prepare_pathmnist(npz_path: Path, frames: dict[str, pd.DataFrame], cache_dir: Path, size: int, log=print) -> None:
    todo = {s: f for s, f in frames.items() if not _is_built(cache_dir, "pathmnist", s, f)}
    if not todo:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    mm = {s: open_memmap(_paths(cache_dir, "pathmnist", s)[0], mode="w+", dtype=np.uint8,
                         shape=(len(f), size, size, 3)) for s, f in todo.items()}
    targets: dict[str, list] = {}                      # source key -> [(src_index, split, position)]
    for s, f in todo.items():
        key = f["uid"].str.split("/").str[1]
        for pos, (k, i) in enumerate(zip(key, f["src_index"])):
            targets.setdefault(k, []).append((int(i), s, pos))
    z = np.load(npz_path)
    for k, items in targets.items():
        items.sort()
        idx = np.array([t[0] for t in items])
        lab = z[f"{k}_labels"].reshape(-1)
        for start, chunk in iter_npz_chunks(npz_path, f"{k}_images", 1024):
            a, b = np.searchsorted(idx, [start, start + len(chunk)])
            for j in range(a, b):
                _, s, pos = items[j]
                mm[s][pos] = chunk[idx[j] - start]
        for s, f in todo.items():                      # labels in the frozen manifest must match the source
            sel = [(t[0], t[2]) for t in items if t[1] == s]
            if sel:
                src_i = np.array([t[0] for t in sel]); pos = np.array([t[1] for t in sel])
                assert (lab[src_i] == f["label"].to_numpy()[pos]).all(), f"label mismatch in {s} ({k})"
        log(f"  cached source '{k}': {len(idx)} images")
    for s, f in todo.items():
        mm[s].flush()
        _paths(cache_dir, "pathmnist", s)[1].write_text(json.dumps({"n": len(f), "uid_hash": _uid_hash(f)}))
    del mm


def prepare_camelyon(npz_path: Path, frames: dict[str, pd.DataFrame], cache_dir: Path, size: int, log=print) -> None:
    todo = {s: f for s, f in frames.items() if not _is_built(cache_dir, "camelyon17", s, f)}
    if not todo:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(npz_path)
    imgs, uid = z["images"], z["uid"].astype(str)
    row = {u: i for i, u in enumerate(uid)}
    for s, f in todo.items():
        mm = open_memmap(_paths(cache_dir, "camelyon17", s)[0], mode="w+", dtype=np.uint8, shape=(len(f), size, size, 3))
        idx = np.array([row[u] for u in f["uid"]])
        for a in range(0, len(idx), 4096):
            mm[a:a + 4096] = imgs[idx[a:a + 4096]]
        mm.flush()
        _paths(cache_dir, "camelyon17", s)[1].write_text(json.dumps({"n": len(f), "uid_hash": _uid_hash(f)}))
        log(f"  cached camelyon17 {s}: {len(f)} patches")


def load_split(cache_dir: Path, dataset: str, split: str, frame: pd.DataFrame) -> SplitData:
    assert _is_built(cache_dir, dataset, split, frame), f"cache for {dataset}/{split} missing or stale"
    imgs = np.load(_paths(cache_dir, dataset, split)[0], mmap_mode="r")
    return SplitData(split, imgs, frame["label"].to_numpy(np.int64), frame["uid"].to_numpy(dtype=str),
                     frame["patient"].to_numpy(np.int64))


class RowReader:
    """Random-access rows of a uint8 .npy via explicit reads (no memory mapping).

    A np.memmap keeps touched pages in the process working set, which on Windows shrinks 'available RAM'
    by up to the file size; plain reads go through the reclaimable OS file cache instead."""

    def __init__(self, path: Path):
        arr = np.load(path, mmap_mode="r")
        self.path, self.shape, self.offset = Path(path), tuple(arr.shape), int(arr.offset)
        self.row_bytes = int(np.prod(self.shape[1:]))
        del arr
        self._f = None

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.row_bytes

    def __getitem__(self, idx) -> np.ndarray:
        if self._f is None:
            self._f = open(self.path, "rb", buffering=0)
        idx = np.atleast_1d(np.asarray(idx))
        out = np.empty((len(idx),) + self.shape[1:], dtype=np.uint8)
        for j, i in enumerate(idx):
            self._f.seek(self.offset + int(i) * self.row_bytes)
            self._f.readinto(memoryview(out[j]).cast("B"))
        return out


def split_file(cache_dir: Path, dataset: str, split: str) -> Path:
    return _paths(cache_dir, dataset, split)[0]


def to_device(imgs, device, chunk: int = 4096):
    """Copy a uint8 memmap to a preallocated GPU tensor chunk by chunk (never holds more than one chunk in RAM)."""
    import torch
    out = torch.empty(imgs.shape, dtype=torch.uint8, device=device)
    for a in range(0, len(imgs), chunk):
        out[a:a + chunk] = torch.from_numpy(np.array(imgs[a:a + chunk])).to(device)
    return out

