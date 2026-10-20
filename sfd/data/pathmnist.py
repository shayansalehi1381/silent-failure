"""PathMNIST (MedMNIST v2) loading as uint8 tensors kept in RAM."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_npz_rows(path: str | Path, key: str, indices, chunk_rows: int = 1024) -> dict[int, np.ndarray]:
    """Rows `indices` of array `key` in an .npz, streaming the member so peak RAM is one chunk
    (np.load would materialise the whole array, 4.4 GB for PathMNIST 128px train)."""
    import zipfile

    want = {int(i) for i in indices}
    out: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(f)
        assert not fortran
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk_rows):
            n = min(chunk_rows, shape[0] - start)
            buf = np.frombuffer(f.read(n * row_bytes), dtype=dtype).reshape((n,) + tuple(shape[1:]))
            for i in want:
                if start <= i < start + n:
                    out[i] = buf[i - start].copy()
            if len(out) == len(want):
                break
    return out


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
