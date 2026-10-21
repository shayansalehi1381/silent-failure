"""Family A: MedMNIST-C v0.1.0 corruptions applied at 224 px, then resized to 128 px.

MedMNIST-C's `corruptions/filter.py` imports `wand` (ImageMagick) at module level although only `motion_blur` uses
it. A stub `wand` module is installed before the import, so every other corruption runs exactly as published;
`motion_blur` raises until ImageMagick is installed (kept as `pending` in the config).
Determinism: every (corruption, severity, uid) gets its own seed for Python's `random` and NumPy's legacy RNG
(the generators MedMNIST-C uses), so results do not depend on processing order or worker count.
Images are read/written with explicit file I/O (no memory mapping) to keep the Windows working set small.
"""
from __future__ import annotations

import hashlib
import random
import sys
import types
from pathlib import Path

import numpy as np

RESIZERS = ("pil_bicubic", "pil_bilinear", "pil_lanczos", "pil_box", "cv2_area")


def _numpy2_compat() -> None:
    """MedMNIST-C's motion_blur calls np.fromstring(bytes, uint8), whose binary mode NumPy 2 removed. NumPy 1.x
    binary mode returned a copy of the buffer, so frombuffer(...).copy() is the exact equivalent (text mode unchanged)."""
    if getattr(np.fromstring, "_sfd_compat", False):
        return
    orig = np.fromstring

    def fromstring(s, dtype=float, count=-1, sep="", **kw):
        if sep == "":
            return np.frombuffer(s, dtype=dtype, count=count).copy()
        return orig(s, dtype=dtype, count=count, sep=sep, **kw)
    fromstring._sfd_compat = True
    np.fromstring = fromstring


def _stub_wand() -> None:
    """Only if the real Wand/ImageMagick is unavailable: a stub so the other corruptions still import."""
    _numpy2_compat()
    if "wand" in sys.modules:
        return
    try:
        import wand.image  # noqa: F401  (real ImageMagick binding present)
        return
    except Exception:
        pass

    class _NoWand:
        def __init__(self, *a, **k):
            raise RuntimeError("ImageMagick/wand not installed: motion_blur is unavailable")

    for n in ("wand", "wand.image", "wand.api"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["wand.image"].Image = _NoWand
    sys.modules["wand.api"].library = None


def corruptors(dataset: str = "pathmnist") -> dict:
    _stub_wand()
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    return CORRUPTIONS_DS[dataset]


def image_seed(base: int, corruption: str, severity: int, uid: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{corruption}|{severity}|{uid}".encode()).digest()[:4], "little")


def resize128(img: np.ndarray, method: str) -> np.ndarray:
    """224 -> 128 with the named interpolation (uint8 RGB in, uint8 RGB out)."""
    if method == "cv2_area":
        import cv2
        return cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    from PIL import Image
    f = {"pil_bicubic": Image.BICUBIC, "pil_bilinear": Image.BILINEAR, "pil_lanczos": Image.LANCZOS,
         "pil_box": Image.BOX}[method]
    return np.asarray(Image.fromarray(img).resize((128, 128), f))


def corrupt_one(corruptor, img224: np.ndarray, severity: int, seed: int) -> np.ndarray:
    """Apply one MedMNIST-C corruption (severity 1..5) to a 224x224 RGB uint8 image."""
    from PIL import Image
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    out = corruptor.apply(Image.fromarray(img224).convert("RGB"), severity - 1)   # MedMNIST-C indexes 0..4
    out = np.asarray(out)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8, (out.shape, out.dtype)
    return out


class RowIO:
    """Explicit-read/write access to rows of an existing uint8 .npy file (no memory mapping)."""

    def __init__(self, path: Path):
        a = np.load(path, mmap_mode="r")
        self.path, self.shape, self.offset = Path(path), tuple(a.shape), int(a.offset)
        self.row = int(np.prod(self.shape[1:]))
        del a

    def read(self, i0: int, i1: int) -> np.ndarray:
        with open(self.path, "rb") as f:
            f.seek(self.offset + i0 * self.row)
            return np.frombuffer(f.read((i1 - i0) * self.row), np.uint8).reshape((i1 - i0,) + self.shape[1:]).copy()

    def write(self, i0: int, arr: np.ndarray) -> None:
        assert arr.dtype == np.uint8 and arr.shape[1:] == self.shape[1:]
        with open(self.path, "r+b") as f:
            f.seek(self.offset + i0 * self.row)
            f.write(np.ascontiguousarray(arr).tobytes())


def task(args) -> tuple[int, int]:
    """Worker: corrupt rows [i0, i1) of the clean 224 sample for one (corruption, severity) and write them."""
    clean_path, out_path, corruption, severity, i0, i1, uids, base_seed, method = args
    c = corruptors()[corruption]
    src, dst = RowIO(Path(clean_path)), RowIO(Path(out_path))
    imgs = src.read(i0, i1)
    out = np.empty((i1 - i0, 128, 128, 3), np.uint8)
    for k in range(i1 - i0):
        out[k] = resize128(corrupt_one(c, imgs[k], severity, image_seed(base_seed, corruption, severity, uids[k])), method)
    dst.write(i0, out)
    return i0, i1
