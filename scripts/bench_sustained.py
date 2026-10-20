"""Phase 0 sustained benchmark: continuous ResNet-18 training on the full PathMNIST 128px train set.

Runs for `bench.sustained_seconds` (no early stopping, epochs wrap around), AMP + channels_last,
GPU augmentation, uint8 data in RAM. A background thread samples nvidia-smi every
`bench.sample_period_s` seconds; the test is stopped if GPU temperature exceeds `bench.temp_stop_c`.
Writes outputs/bench_sustained.json. Results are measured, never estimated, except the labelled
extrapolations (epoch time, 5-seed totals), which use the throughput of the LAST minute.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import gpu_augment  # noqa: E402
from env_record import _ram_gb  # noqa: E402
from sfd.config import ROOT, load_config, resolve, set_seed  # noqa: E402
from sfd.data.pathmnist import load_pathmnist  # noqa: E402

# bit masks of clocks_throttle_reasons (NVML); 0x1 = idle and 0x2 = app clocks are NOT throttling
THROTTLE_BITS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal_slowdown",
                 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake"}
FIELDS = ["temperature.gpu", "power.draw", "clocks.sm", "utilization.gpu", "memory.used"]


def _query_fields() -> list[str]:
    """Throttle-reason field was renamed in newer drivers; use whichever this driver accepts."""
    for name in ("clocks_throttle_reasons.active", "clocks_event_reasons.active"):
        try:
            subprocess.check_output(["nvidia-smi", f"--query-gpu={name}", "--format=csv,noheader"],
                                    text=True, stderr=subprocess.DEVNULL)
            return FIELDS + [name]
        except Exception:
            continue
    return FIELDS


class Sampler(threading.Thread):
    def __init__(self, period: float, temp_stop: float):
        super().__init__(daemon=True)
        self.period, self.temp_stop = period, temp_stop
        self.fields = _query_fields()
        self.rows: list[dict] = []
        self.halt, self.overheated = threading.Event(), threading.Event()
        self.t0 = time.time()

    def run(self):
        q = ",".join(self.fields)
        while not self.halt.is_set():
            try:
                o = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                    text=True).strip().split(",")
                row = {"t": round(time.time() - self.t0, 1)}
                for f, v in zip(self.fields, o):
                    v = v.strip()
                    row[f] = int(v, 16) if v.startswith("0x") else float(v)
                self.rows.append(row)
                if row["temperature.gpu"] > self.temp_stop:
                    self.overheated.set()
            except Exception as e:  # keep sampling; a missed sample is reported via n_samples
                print(f"[sampler] {e}")
            self.halt.wait(self.period)

    def stop(self):
        self.halt.set()
        self.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--seconds", type=float, default=None, help="override bench.sustained_seconds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    b = cfg["bench"]
    seconds = a.seconds or b["sustained_seconds"]
    set_seed(cfg["seed"])
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    ram = _ram_gb()
    print(f"free RAM before loading: {ram.get('available')} GB of {ram.get('total')} GB")
    if ram.get("available", 99) < b["min_free_ram_gb"]:
        print(f"WARNING: free RAM below {b['min_free_ram_gb']} GB; close other programs "
              f"(the uint8 train set needs ~4.4 GB)")

    imgs, labels = load_pathmnist(resolve(cfg, "data_dir"), cfg["data"]["pathmnist_size"],
                                  splits=("train",), download=False)["train"]
    n_full = len(imgs)
    ram_after = _ram_gb()
    print(f"train: {imgs.shape} {imgs.dtype}; free RAM after loading: {ram_after.get('available')} GB")

    bs, accum = cfg["train"]["batch_size"], cfg["train"]["grad_accum"]
    micro = bs // accum
    model = torchvision.models.resnet18(num_classes=9).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x_all, y_all = torch.from_numpy(imgs), torch.from_numpy(labels)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model.train()

    def step(idx):
        opt.zero_grad(set_to_none=True)
        for j in range(accum):
            sub = idx[j * micro:(j + 1) * micro]
            x = x_all[sub].to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            x = ((gpu_augment(x) - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y_all[sub].to(dev)) / accum
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # warm-up (cudnn autotune), excluded from timing
    for _ in range(3):
        step(torch.randperm(n_full)[:bs])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sampler = Sampler(b["sample_period_s"], b["temp_stop_c"])
    sampler.start()
    t0 = time.time()
    marks = [(0.0, 0)]            # (elapsed seconds, images processed), synced every 50 steps
    done, epochs_started, stopped_for_heat = 0, 0, False
    while time.time() - t0 < seconds and not stopped_for_heat:
        perm = torch.randperm(n_full)
        epochs_started += 1
        for i in range(0, n_full - bs + 1, bs):
            step(perm[i:i + bs])
            done += bs
            if (done // bs) % 50 == 0:
                torch.cuda.synchronize()
                marks.append((time.time() - t0, done))
                if sampler.overheated.is_set():
                    stopped_for_heat = True
                    print(f"STOP: GPU temperature exceeded {b['temp_stop_c']} C")
                    break
                if time.time() - t0 >= seconds:
                    break
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    marks.append((elapsed, done))
    sampler.stop()

    t = np.array([m[0] for m in marks])
    n = np.array([m[1] for m in marks], dtype=float)

    def rate(lo: float, hi: float) -> float | None:
        if hi > t[-1] + 1e-6 or lo < 0:
            return None
        return round(float((np.interp(hi, t, n) - np.interp(lo, t, n)) / (hi - lo)), 1)

    first_min = rate(0, 60)
    last_min = rate(elapsed - 60, elapsed) if elapsed >= 120 else None
    overall = round(done / elapsed, 1)
    per_minute = [rate(m * 60, (m + 1) * 60) for m in range(int(elapsed // 60))]

    rows = sampler.rows
    col = lambda k: np.array([r[k] for r in rows if k in r], dtype=float)  # noqa: E731
    temps = col("temperature.gpu")
    tkey = next((k for k in sampler.fields if k.endswith("reasons.active")), None)
    reasons = sorted({int(r[tkey]) for r in rows}) if tkey else []
    throttle_flags = sorted({name for r in rows if tkey for bit, name in THROTTLE_BITS.items()
                             if int(r[tkey]) & bit})
    # extrapolations use the last-minute rate when available (the sustained, not the cold, speed)
    ref = last_min or overall
    epoch_min = n_full / ref / 60
    res = {
        "seconds_run": round(elapsed, 1), "images_processed": int(done),
        "epochs_started": epochs_started, "batch": bs, "grad_accum": accum,
        "stopped_for_heat": stopped_for_heat,
        "images_per_sec": {"first_minute": first_min, "last_minute": last_min,
                           "overall": overall, "per_minute": per_minute},
        "gpu": {
            "temp_c_max": float(temps.max()) if temps.size else None,
            "temp_c_mean": round(float(temps.mean()), 1) if temps.size else None,
            "power_w_max": float(col("power.draw").max()) if rows else None,
            "power_w_mean": round(float(col("power.draw").mean()), 1) if rows else None,
            "sm_clock_mhz_max": float(col("clocks.sm").max()) if rows else None,
            "sm_clock_mhz_min": float(col("clocks.sm").min()) if rows else None,
            "util_pct_mean": round(float(col("utilization.gpu").mean()), 1) if rows else None,
            "vram_used_mib_max": float(col("memory.used").max()) if rows else None,
            "throttle_reason_masks_seen": [hex(r) for r in reasons],
            "throttling_occurred": bool(throttle_flags),
            "throttle_flags": throttle_flags,
            "n_samples": len(rows), "sample_period_s": b["sample_period_s"],
        },
        "torch_peak_vram_gb": {
            "allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2)},
        "ram_gb": {"before_load": ram, "after_load": ram_after},
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1),
        "n_train_full": n_full,
        "extrapolation_from": "last_minute" if last_min else "overall",
        "epoch_min_full_train": round(epoch_min, 2),
        "estimates_min_5seeds": {f"{e}_epochs": round(5 * e * epoch_min, 0) for e in (10, 12)},
        "samples": rows,
    }
    out = Path(a.out) if a.out else ROOT / "outputs" / "bench_sustained.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=2))
    print(f"saved -> {out}")
    if res["gpu"]["temp_c_max"] and res["gpu"]["temp_c_max"] > b["temp_stop_c"]:
        print(f"WARNING: GPU reached {res['gpu']['temp_c_max']} C (> {b['temp_stop_c']} C)")


if __name__ == "__main__":
    main()
