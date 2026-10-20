"""Phase 0 benchmark: ResNet-18 (9 classes), one pass over N PathMNIST 128px train images.

Data stays in RAM as uint8; batches are moved to GPU and augmented there (flip/rot90).
AMP (fp16) + channels_last. Falls back from batch 128 to 64 + grad accumulation on OOM.
Samples GPU temperature/power via nvidia-smi in a background thread.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfd.config import ROOT, load_config, resolve, set_seed  # noqa: E402
from sfd.data.pathmnist import load_pathmnist  # noqa: E402


class GpuSampler(threading.Thread):
    def __init__(self, period: float = 1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                o = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], text=True).strip().split(",")
                self.samples.append([float(x) for x in o])
            except Exception:
                pass
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join()

    def summary(self):
        a = np.array(self.samples)
        if a.size == 0:
            return {}
        return {"temp_c_max": a[:, 0].max(), "temp_c_mean": round(a[:, 0].mean(), 1),
                "power_w_max": a[:, 1].max(), "power_w_mean": round(a[:, 1].mean(), 1),
                "util_pct_mean": round(a[:, 2].mean(), 1), "vram_used_mib_max": a[:, 3].max(),
                "n_samples": len(a)}


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """x: float NCHW on GPU. Random horizontal/vertical flip + rot90 per batch."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1).item() < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), (2, 3))


def run(imgs: np.ndarray, labels: np.ndarray, bs: int, accum: int, dev) -> float:
    model = torchvision.models.resnet18(num_classes=9).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x_all = torch.from_numpy(imgs)           # uint8 NHWC, CPU/RAM
    y_all = torch.from_numpy(labels)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model.train()
    n = len(x_all)
    perm = torch.randperm(n)
    micro = bs // accum
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, n - bs + 1, bs):
        idx = perm[i:i + bs]
        opt.zero_grad(set_to_none=True)
        for j in range(accum):
            sub = idx[j * micro:(j + 1) * micro]
            x = x_all[sub].to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            x = ((gpu_augment(x) - mean) / std).contiguous(memory_format=torch.channels_last)
            y = y_all[sub].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    data = load_pathmnist(resolve(cfg, "data_dir"), cfg["data"]["pathmnist_size"])
    for s, (x, y) in data.items():
        print(f"{s:5s}: {x.shape} dtype={x.dtype}  class counts={np.bincount(y, minlength=9).tolist()}")
    n_full = len(data["train"][0])
    n = cfg["bench"]["n_images"]
    rng = np.random.default_rng(cfg["seed"])
    sel = rng.choice(n_full, n, replace=False)
    imgs, labels = data["train"][0][sel], data["train"][1][sel]

    # warm-up (cudnn autotune, kernels) on 2 batches, excluded from timing
    run(imgs[:256], labels[:256], 128, 1, dev)
    torch.cuda.reset_peak_memory_stats()

    sampler = GpuSampler()
    sampler.start()
    bs, accum = 128, 1
    try:
        secs = run(imgs, labels, bs, accum, dev)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        bs, accum = 128, 2   # effective 128 = 2 x 64
        print("OOM at micro-batch 128 -> falling back to 64 x accumulation 2")
        secs = run(imgs, labels, bs, accum, dev)
    sampler.stop()

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    img_s = n / secs
    epoch_min = n_full / img_s / 60
    res = {
        "n_images": n, "batch": bs, "grad_accum": accum, "seconds": round(secs, 1),
        "images_per_sec": round(img_s, 1), "n_train_full": n_full,
        "epoch_min_full_train": round(epoch_min, 2),
        "peak_vram_allocated_gb": round(peak_alloc, 2), "peak_vram_reserved_gb": round(peak_res, 2),
        "gpu_sampled": sampler.summary(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1),
        "estimates_min_5seeds": {f"{e}_epochs": round(5 * e * epoch_min, 0) for e in (10, 20, 30)},
    }
    print(json.dumps(res, indent=2))
    out = Path(a.out) if a.out else ROOT / "outputs" / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    t = res["gpu_sampled"].get("temp_c_max", 0)
    limit = cfg["bench"].get("temp_stop_c", 99)
    if t > limit:
        print(f"WARNING: GPU reached {t} C (> {limit} C)")


if __name__ == "__main__":
    main()
