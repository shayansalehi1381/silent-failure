# Silent failure in medical image classifiers under distribution shift

Do representation drift and Grad-CAM explanation drift detect *silent failures*
(wrong prediction with high calibrated confidence) beyond calibrated uncertainty, and do
they reduce the residual silent-failure rate under a fixed human-review budget?

Design: `ROADMAP_Shayan.md` (section 10 takes precedence). Project rules: `CLAUDE.md`.

## Setup (Windows 11, conda env `sfd`, Python 3.11)
```bash
conda activate sfd
# 1. CUDA 12.8 build of torch first (not part of requirements.txt)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# 2. everything else, pinned
pip install -r requirements.txt
```

## How to run (Phase 0)
```bash
python scripts/env_record.py --config configs/full.yaml     # outputs/env_record.json
python scripts/bench.py --config configs/smoke.yaml         # short benchmark -> outputs/bench.json
python scripts/bench_sustained.py --config configs/full.yaml  # sustained benchmark -> outputs/bench_sustained.json
python scripts/run_all.py --config configs/smoke.yaml       # pipeline smoke test; add --resume
python -m pytest                                            # unit tests (~15 s)
```
The first call that needs PathMNIST downloads `data/pathmnist_128.npz` (4.26 GB, git-ignored).

## Fixed decisions
- **Datasets / versions** (agreed with the group, do not change without asking): PathMNIST 128x128
  (MedMNIST v2), MedMNIST-C v0.1.0 (corruptions applied at 224, then resized to 128),
  Camelyon17-WILDS v1.0 at 96x96.
- **Model:** ResNet-18 trained **from scratch** (no ImageNet weights, to avoid a pretraining
  confound), 5 seeds, max 12 epochs, early stopping on source val loss (patience 3),
  temperature scaling on source val. Config: `configs/full.yaml`.
- **Sub-dataset policy:** PathMNIST uses the FULL train set (~90k); its official val and test splits
  are untouched. Camelyon17 uses a stratified, hospital/slide-aware sample with a fixed seed saved as
  `subset_manifest.parquet`; its sizes are config parameters, still TBD. MedMNIST-C corruptions are applied
  only to the subset's val/test images. The report will include a "full vs used size" table.

## Hardware notes
RTX 3060 Laptop GPU, 6 GB VRAM, 15.6 GB RAM (often mostly used), Windows. No DataLoader workers:
uint8 data stays in RAM (PathMNIST 128px train is ~4.4 GB), augmentation runs on the GPU,
`torch.autocast` fp16 + `channels_last`, batch 128 (fallback 64 x 2 accumulation). The laptop runs hot:
keep it on AC power with good airflow; the sustained benchmark aborts at 99 C (card slowdown temp 102 C minus 3 C; shutdown 105 C).

## Phase 0 benchmark results (measured, ResNet-18, batch 128, AMP, channels_last)
| Benchmark | Images/s | Epoch (90k) | Peak VRAM (alloc/reserved) | GPU temp max | Throttling |
|---|---|---|---|---|---|
| Short (`bench.json`; 2000 images, 5 GPU samples) | 475 | 3.16 min | 0.62 / 0.73 GB | 76 C | not recorded |
| Sustained (`bench_sustained.json`; 240.8 s, 102,400 images, 48 samples at 5 s) | first min 444.3, last min 406.1 (per minute: 444, 438, 413, 406) | 3.69 min (from last minute) | 0.62 / 0.72 GB | 87 C (mean 84.1) | SW power cap and **SW thermal slowdown** (no HW thermal) |

Sustained run details: started at 60 C after a 211 s cooldown from 77 C; abort threshold 99 C (never reached).
From about 90 s the GPU sat at its 87 C target temperature, SW thermal slowdown (0x20) was active for the rest
of the run, the SM clock fell from ~1790 to ~1220-1500 MHz and power from ~78 W to ~55 W, costing ~9 % throughput
(444 -> 406 img/s, still slowly declining in minute 4). Training is safe at this operating point (15 C under the card's
slowdown temperature) but ~9 % slower than a cold start. Estimated GPU time for 5 seeds x 12 epochs (maximum, early
stopping may shorten it): ~222 min; x 10 epochs: ~185 min. Free RAM: 7.2 GB before, 3.1 GB after loading the train set.
Card limits (`nvidia-smi -q -d TEMPERATURE`): target 87 C, slowdown 102 C, max operating/shutdown 105 C.

## Status
| Phase | Content | Status |
|---|---|---|
| 0 | Repo, env, requirements, configs, PathMNIST loader, benchmarks, tests, smoke | done |
| 1-9 | Stages 1-9 of the roadmap | not started |
