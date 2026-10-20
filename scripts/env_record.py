"""Save environment info (versions, GPU, RAM, disk, git commit, seed) to env_record.json."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfd.config import ROOT, load_config, resolve, set_seed  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


def _ram_gb() -> dict:
    """Total/available physical RAM (Windows, via ctypes; avoids a psutil dependency)."""
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = MEMSTATUS()
        s.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return {"total": round(s.ullTotalPhys / 2**30, 2), "available": round(s.ullAvailPhys / 2**30, 2)}
    except Exception:
        return {}


def collect(seed: int, data_dir: Path) -> dict:
    import torch

    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": _nvidia_smi(),
        "ram_gb": _ram_gb(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1),
        "git_commit": _git_commit(),
        "seed": seed,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 2),
                      "capability": f"{p.major}.{p.minor}"}
    # the pytorch-grad-cam library is published on PyPI as "grad-cam"
    for pkg in ["torchvision", "numpy", "pillow", "pandas", "scikit-learn", "scipy", "medmnist",
                "wilds", "grad-cam", "statsmodels", "pyarrow", "matplotlib", "seaborn", "tqdm",
                "pyyaml", "jupyter", "pytest"]:
        try:
            from importlib.metadata import version

            rec.setdefault("packages", {})[pkg] = version(pkg)
        except Exception:
            rec.setdefault("packages", {})[pkg] = None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default=None, help="output json (default outputs/env_record.json)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    rec = collect(cfg["seed"], resolve(cfg, "data_dir"))
    out = Path(a.out) if a.out else ROOT / "outputs" / "env_record.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
