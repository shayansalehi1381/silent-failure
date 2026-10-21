"""ResNet-18 source-classifier training (ImageNet-pretrained), GPU-resident uint8 data, resumable per epoch.

Augmentation is ONLY random horizontal/vertical flips and random 90-degree rotations (per sample, on the
GPU). No colour, brightness, contrast, blur or noise augmentation: those would pre-train robustness to
the MedMNIST-C corruptions (shift family A) and bias the study.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from sfd.config import set_seed
from sfd.sysinfo import ram_gb

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- safety ----------
def read_slowdown_temp() -> float | None:
    """GPU slowdown temperature reported by `nvidia-smi -q -d TEMPERATURE` (None if not reported)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "-q", "-d", "TEMPERATURE"], text=True)
        m = re.search(r"GPU Slowdown Temp\s*:\s*(\d+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def gpu_temp() -> float | None:
    try:
        return float(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                              "--format=csv,noheader,nounits"], text=True).strip())
    except Exception:
        return None


class GuardStop(RuntimeError):
    """Raised when training must stop and be reported (low RAM, or cooling did not help)."""


class GpuGuard:
    """Pauses while the GPU is at/above slowdown-margin; raises GuardStop if free RAM < min_free_ram_gb."""

    def __init__(self, margin_c: float = 3.0, min_free_ram_gb: float = 2.0, pause_s: float = 20,
                 max_pause_s: float = 900, log=print):
        slow = read_slowdown_temp()
        self.limit = (slow - margin_c) if slow else 85.0
        self.min_free, self.pause_s, self.max_pause_s, self.log = min_free_ram_gb, pause_s, max_pause_s, log
        self.log(f"[guard] slowdown temp {slow} C -> pause at >= {self.limit} C; stop if free RAM < {min_free_ram_gb} GB")

    def check(self) -> None:
        free = ram_gb().get("available")
        if free is not None and free < self.min_free:
            raise GuardStop(f"free RAM {free} GB < {self.min_free} GB")
        t, waited = gpu_temp(), 0.0
        while t is not None and t >= self.limit:
            self.log(f"[guard] GPU {t} C >= {self.limit} C: pausing {self.pause_s:.0f} s")
            time.sleep(self.pause_s)
            waited += self.pause_s
            if waited > self.max_pause_s:
                raise GuardStop(f"GPU stayed >= {self.limit} C for {waited:.0f} s")
            t = gpu_temp()


# --------------------------------------------------------------------------- model -----------
class Net(nn.Module):
    """torchvision ResNet-18 with a fresh linear head; `features` returns the 512-d penultimate vector."""

    def __init__(self, n_classes: int, pretrained: bool = True, weights: str = "IMAGENET1K_V1"):
        super().__init__()
        w = getattr(torchvision.models.ResNet18_Weights, weights) if pretrained else None
        self.net = torchvision.models.resnet18(weights=w)
        self.net.fc = nn.Linear(512, n_classes)

    def features(self, x):
        m = self.net
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return torch.flatten(m.avgpool(x), 1)

    def forward(self, x):
        return self.net.fc(self.features(x))


def augment(x: torch.Tensor) -> torch.Tensor:
    """Per-sample random h-flip, v-flip and rot90 (k in 0..3). x: N x C x S x S float on the GPU (square)."""
    n, dev = x.shape[0], x.device
    f = torch.rand(n, 2, device=dev) < 0.5
    x = torch.where(f[:, 0].view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(f[:, 1].view(-1, 1, 1, 1), x.flip(2), x)
    k = torch.randint(0, 4, (n,), device=dev)
    for r in (1, 2, 3):
        m = k == r
        if m.any():
            x[m] = torch.rot90(x[m], r, (2, 3))
    return x


def _prep(u8: torch.Tensor, dev, mean, std) -> torch.Tensor:
    x = u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return x


def _norm(x, mean, std):
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model: Net, images, dev, bs: int = 512, amp: bool = False):
    """logits (float32) and penultimate features (float16) for uint8 NHWC images (memmap, ndarray or tensor)."""
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    model.eval()
    L, F = [], []
    for a in range(0, len(images), bs):
        chunk = images[a:a + bs]
        u8 = chunk if torch.is_tensor(chunk) else torch.from_numpy(np.array(chunk))
        x = _norm(_prep(u8, dev, mean, std), mean, std)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            f = model.features(x)
            lo = model.net.fc(f)
        L.append(lo.float().cpu())
        F.append(f.half().cpu())
    return torch.cat(L).numpy(), torch.cat(F).numpy()


# --------------------------------------------------------------------------- training ---------
def train_one(cfg: dict, tr, va, n_classes: int, seed: int, ckpt_dir: Path, log_path: Path, guard: GpuGuard,
              dev, log=print, pretrained: bool | None = None) -> dict:
    """Train with early stopping on `va` loss. `tr.images` must be a uint8 tensor on the GPU.
    Resumable: `last.pt` is written after every epoch; `best.pt` holds the best-val_loss weights."""
    t_cfg, m_cfg = cfg["train"], cfg["model"]
    o = t_cfg["optimizer"]
    assert o["name"] == "sgd" and o.get("label_smoothing", 0.0) == 0.0
    bs, accum = t_cfg["batch_size"], t_cfg["grad_accum"]
    assert accum == 1, "grad_accum > 1 not needed at batch 128 (peak VRAM < 1 GB for the model)"
    epochs, patience = t_cfg["epochs"], t_cfg["early_stopping"]["patience"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    model = Net(n_classes, m_cfg["pretrained"] if pretrained is None else pretrained,
                m_cfg.get("weights", "IMAGENET1K_V1")).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    n = len(tr.images)
    spe = n // bs
    total = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=t_cfg["amp"])
    crit = nn.CrossEntropyLoss()
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    y_all = torch.from_numpy(tr.labels.copy()).to(dev)
    on_gpu = torch.is_tensor(tr.images)        # else: np.memmap on the SSD, batches gathered on the CPU
    st = {"epoch": 0, "best_val": float("inf"), "best_epoch": -1, "bad": 0, "history": []}
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); st = ck["state"]
        log(f"  resumed from epoch {st['epoch']} (best {st['best_val']:.4f} at {st['best_epoch']})")
    while st["epoch"] < epochs and st["bad"] < patience:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        perm = torch.randperm(n, device=dev if on_gpu else "cpu")
        run = torch.zeros((), device=dev)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            if on_gpu:
                u8 = tr.images[idx]
            else:                                  # sorted indices: sequential-ish reads from the memmap
                idx = torch.sort(idx).values
                u8 = torch.from_numpy(tr.images[idx.numpy()])
                idx = idx.to(dev)
            x = _norm(augment(_prep(u8, dev, mean, std)), mean, std)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=t_cfg["amp"]):
                loss = crit(model(x), y_all[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.detach()
            if i % 100 == 99:
                guard.check()
        lo, _ = predict(model, va.images, dev, amp=t_cfg["amp"])
        vloss = float(nn.functional.cross_entropy(torch.from_numpy(lo), torch.from_numpy(va.labels)))
        vacc = float((lo.argmax(1) == va.labels).mean())
        if not (math.isfinite(vloss) and math.isfinite(float(run))):
            raise GuardStop(f"non-finite loss at epoch {st['epoch']}: train {float(run)}, val {vloss}")
        rec = {"epoch": st["epoch"] + 1, "train_loss": round(float(run) / spe, 5), "val_loss": round(vloss, 5),
               "val_acc": round(vacc, 5), "epoch_s": round(time.time() - t0, 1), "gpu_temp_c": gpu_temp(),
               "free_ram_gb": ram_gb().get("available"), "lr": round(sched.get_last_lr()[0], 6),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        st["history"].append(rec)
        st["epoch"] += 1
        if vloss < st["best_val"] - 1e-6:
            st["best_val"], st["best_epoch"], st["bad"] = vloss, st["epoch"], 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            st["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": st}, last)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"  seed {seed} epoch {rec['epoch']}: {rec}")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=dev))
    return {"model": model, "best_epoch": st["best_epoch"], "best_val_loss": st["best_val"],
            "epochs_run": st["epoch"], "history": st["history"]}

