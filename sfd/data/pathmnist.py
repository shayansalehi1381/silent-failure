"""PathMNIST (MedMNIST v2) loading as uint8 tensors kept in RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pathmnist(data_dir: str | Path, size: int = 128, splits=("train", "val", "test"),
                   download: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {split: (images uint8 NxHxWx3, labels int64 N)}."""
    import medmnist
    from medmnist import INFO

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, INFO["pathmnist"]["python_class"])
    out = {}
    for s in splits:
        ds = cls(split=s, size=size, root=str(data_dir), download=download)
        out[s] = (ds.imgs, ds.labels.reshape(-1).astype(np.int64))
    return out
