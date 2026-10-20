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
python scripts/run_all.py --config configs/full.yaml        # stage 0 + stage 1 (outputs/stage1/); add --resume
python -m pytest                                            # unit + integrity tests (~40 s)
```
The first call that needs PathMNIST downloads `data/pathmnist_128.npz` (4.26 GB, git-ignored).

## Stage 1 - data, frozen splits, leakage audit (`python scripts/run_all.py --config configs/full.yaml`)
Code: `sfd/data/` (`stage1.py` orchestrates; `fingerprint`/`dedup.py`, `splits.py`, `camelyon.py`, `remote.py`, `audit.py`).
Outputs in `outputs/stage1/`: `splits.parquet` (frozen, all datasets), `subset_manifest.parquet` (Camelyon subset),
`dedup_report.csv` (threshold sensitivity), `leakage_audit.json`, `table1.csv/.md`, `near_duplicate_pairs.png`,
`camelyon_spatial_overlap*.csv`, `checksums.sha256`, `stage1_report.json` (runtime, RAM, transfer). Heavy intermediate
results are cached in `outputs/stage1/cache/` (git-ignored), keyed by the config, so re-runs are fast.

- **PathMNIST 128px:** pool = official train + val (100,000). Two separate, configurable thresholds on the RMS
  difference of 16x16x3 thumbnails (0 = identical):
  - **within-pool clustering, 0.02** (`pathmnist.dedup.rms_threshold`): connected components of image pairs closer than
    this are near-duplicate clusters and are never split. Conservative on purpose (keeps look-alikes together).
    Cluster-level stratified split 80/5/5/10 into `train`/`val_es`/`val_cal`/`id_test`.
  - **cross-set exclusion from `nct7k_test`, 0.01** (`pathmnist.dedup.cross_exclusion_rms`): the official test
    (CRC-VAL-HE-7K, family X) images closer than this to a pool image are marked `excluded` (kept in `splits.parquet`,
    never used). Result: **0 exclusions, 7,180 images kept**. There are no exact (SHA-256) duplicates anywhere.
  - **Why two thresholds:** at 0.02 the exclusions are 364 images, 363 of them blank BACK tiles (43% of that class in
    family X) that merely look alike at thumbnail scale (see `near_duplicate_pairs.png`); removing them would bias
    family X. The same blank tiles chain into giant clusters when the threshold grows (largest pool cluster 1,067
    images at 0.01, 4,750 at 0.02, 20,278 = 20% of the pool at 0.05), which is why the clustering threshold is
    conservative but not larger.

  | RMS threshold | pool pairs | pool clusters of size > 1 (images in them) | largest cluster | nct7k_test images excluded (per class) |
  |---|---|---|---|---|
  | 0.01 | 840,891 | 43 (3,606) | 1,067 | 0 |
  | 0.02 | 2,690,968 | 43 (7,170) | 4,750 | 364 (BACK 363, ADI 1) |
  | 0.05 | 13,890,805 | 33 (20,487) | 20,278 | 1,195 (BACK 757, mucus 204, ...) |

  Full table (0 - 0.05): `outputs/stage1/dedup_report.csv`; per-class exclusions per threshold: `leakage_audit.json`.
- **Camelyon17-WILDS 96px:** source hospitals 0,3,4 -> **patient-disjoint** `train`/`val_es`/`val_cal`/`id_test`
  (`camelyon.group_by: patient`; sizes in `camelyon.sizes`); hospital 1 -> `ood_c1`, hospital 2 -> `ood_c2`.
  Patient (not slide) is the grouping unit, as in the group partner's setup: a patient can own several slides, and a
  slide-level split put 2 patients into two splits. Source hospitals have 7, 10 and 9 patients (30 slides; 3 patients own
  2-3 slides). The patient-level plan keeps all planned sizes with the same label balance (val_es 31.7% vs 31.2% tumour),
  so nothing was traded off. Labels are balanced within a split as far
  as the patients allow (tumour patches are concentrated in few slides). Non-tissue and dark patches are flagged, not dropped.
  Source: the unofficial HF mirror `wltjr1007/Camelyon17-WILDS` (the official WILDS host has an expired TLS
  certificate; verification is never disabled) or a local WILDS archive (`camelyon.archive_path`). Metadata
  (455,954 rows, 50 slides) was verified against WILDS v1.0 counts using ~36 MiB of range requests; images are streamed
  (~9.9 GiB) only when `camelyon.fetch_images: true`.
- **Camelyon subset audit (real images, 69,000 patches, `data/camelyon17_subset_96.npz`, 1.8 GB, git-ignored):**
  - Download: 9.9 GiB of mirror row groups streamed (8.54 GiB in the final run + 1.4 GiB cached from an interrupted run);
    only the selected patches are kept. 21 shards, TLS verified, resumable via `outputs/stage1/cache/camelyon_parts/`.
  - Near-duplicates (16x16x3 thumbnails, subset self-join, `dedup_report.csv`): at RMS <= 0.02 there are 2,802 pairs and
    29 clusters holding 176 images (0.26%); 0.01: 13 clusters / 114 images; 0.05: chaining, 3,131 images, largest
    cluster 2,843. 55 pairs (RMS <= 0.02) straddle different splits; all 55 join two flagged non-tissue (blank) tiles
    of the same label, i.e. look-alike background, not repeated tissue. They are reported, not asserted.
  - Flags (kept, never dropped; columns `flag_nontissue`, `flag_dark`, `tissue_frac` in `subset_manifest.parquet`):
    772 non-tissue patches (1.1%; 717 non-tumour, 55 tumour) and 93 dark patches, 91 of them in hospital 1 (`ood_c1`),
    a site-specific artefact to keep in mind for family B. Use the flags for sensitivity analyses.
- **Known limitations (Camelyon):**
  - *Label-prior gap.* Tumour patches are concentrated in a few WILDS slides (most slides are at 0-7% tumour, a few at
    60-93%), and patient-disjoint splits cannot fix that. Label-balanced sampling gives train and val_cal 50% tumour,
    but val_es and id_test only ~31% and the OOD splits ~42%. Accuracy/calibration differences between splits include
    this prior shift; rank metrics (AUROC) are less affected.
  - *Few patients per split.* Outside train (17 patients, 20 slides) each source split has only 3 patients, so
    in-distribution test/validation estimates are noisy and dominated by patient effects; bootstrap CIs must
    resample patients (the cluster level), not patches.
  - *Spatial overlap.* In WILDS, on average 69% of patches overlap another patch of the same slide (up to 98%; 38% on
    average in the sampled subset; `camelyon_spatial_overlap*.csv`), which justifies grouping above the patch level.
- **Smoke:** uses a synthetic Camelyon-like fixture (random tiles + WILDS-like metadata) that exercises the code only;
  it is never written to `data/`.
## Fixed decisions
- **Datasets / versions** (agreed with the group, do not change without asking): PathMNIST 128x128
  (MedMNIST v2), MedMNIST-C v0.1.0 (corruptions applied at 224, then resized to 128),
  Camelyon17-WILDS v1.0 at 96x96.
- **Model:** ResNet-18 **initialised from ImageNet** (torchvision `IMAGENET1K_V1`, ~45 MB, downloaded in Phase 2),
  5 seeds, max 8 epochs, early stopping on source `val_es` loss (patience 2), temperature scaling on `val_cal`.
  Optimizer defaults (SGD lr 0.01, cosine, wd 5e-4, no label smoothing) are proposals to be validated on `val_es`,
  never on shifted data. Config: `configs/full.yaml`. This replaces the earlier from-scratch decision because
  ImageNet fine-tuning is standard practice in medical imaging, converges faster, and is comparable with the
  group partner's setup. (The Phase 0 benchmark below was measured with random weights; compute per epoch is identical,
  so 5 seeds x 8 epochs is roughly 5 x 8 x 3.7 min = ~150 min at most, an estimate, not a measurement.)
- **Sub-dataset policy:** PathMNIST uses the FULL train set (~90k); its official val and test splits
  are untouched. Camelyon17 uses a stratified, hospital/patient-aware sample with a fixed seed saved as
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
| 1 | Data, frozen splits, leakage audit | PathMNIST done; Camelyon plan frozen, images pending approval of the ~9.9 GiB transfer |
| 2-9 | Stages 2-9 of the roadmap | not started |

