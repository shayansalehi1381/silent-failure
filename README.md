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
  2-3 slides). The patient-to-split table is **frozen** in `configs/camelyon_assignment_phase1b_rank1.json`
  (patients train/val_es/val_cal/id_test = 10/4/6/6, found by the seeded Phase 1b search, see below). Labels are balanced
  within a split as far as the patients allow. Non-tissue and dark patches are flagged, not dropped.
  Source: the unofficial HF mirror `wltjr1007/Camelyon17-WILDS` (the official WILDS host has an expired TLS
  certificate; verification is never disabled) or a local WILDS archive (`camelyon.archive_path`). Metadata
  (455,954 rows, 50 slides) was verified against WILDS v1.0 counts using ~36 MiB of range requests; images are streamed
  (~9.9 GiB) only when `camelyon.fetch_images: true`.
- **Camelyon subset audit (real images, 69,000 patches, `data/camelyon17_subset_96.npz`, 1.8 GB, git-ignored):**
  - Download: the Phase 1 plan streamed 9.9 GiB of mirror row groups; the Phase 1b rebuild fetched ~7.1 GiB more
    (patches already downloaded are reused per patch). Only the selected patches are kept. TLS verified, resumable via
    `outputs/stage1/cache/camelyon_parts/`. The Phase 1 image subset is archived as
    `data/camelyon17_subset_96_phase1_frozen_plan.npz` (git-ignored); its plan is in git history (tag `phase-1`).
  - Near-duplicates (16x16x3 thumbnails, subset self-join, `dedup_report.csv`): at RMS <= 0.02 there are 2,752 pairs and
    23 clusters holding 146 images (0.21%); 0.01: 12 clusters / 107 images; 0.05: chaining, 2,486 images, largest
    cluster 2,196. 32 pairs (RMS <= 0.02) straddle different splits; they are look-alike background tiles, not repeated
    tissue, and are reported, not asserted.
  - Flags (kept, never dropped; columns `flag_nontissue`, `flag_dark`, `tissue_frac` in `subset_manifest.parquet`):
    797 non-tissue patches (1.2%) and 93 dark patches, 91 of them in hospital 1 (`ood_c1`), a site-specific
    artefact to keep in mind for family B. Use the flags for sensitivity analyses.
- **Why the Camelyon splits changed (Phase 1b).** Calibration (temperature, later the acceptance threshold) is fitted on
  `val_cal` and evaluated on `id_test`/OOD, so a tumour-prior gap between them biases the study. The Phase 1 splits had
  tumour fractions train/val_cal/val_es/id_test = 50/50/31.7/31.0% (a 19-point gap; OOD 42.4%) because tumour patches
  are concentrated in few patients. `scripts/phase1b_patient_search.py` (seeded simulated annealing over all
  patient-disjoint assignments of the 26 source patients, scored with an exact model of the sampler that a test checks
  against the real planner) found assignments with gap <= 0.1 points. Chosen: rank 1 of the strict mode (>= 3 patients per
  split, every hospital in every split including train, each hospital >= 10% of every non-train split). Result:

  | tumour fraction | train | val_es | val_cal | id_test | OOD (hosp. 1 / 2) |
  |---|---|---|---|---|---|
  | Phase 1 splits | 50.0% | 31.7% | 50.0% | 31.0% | 42.2 / 42.5% |
  | Phase 1b rank 1 | 50.0% | 42.4% | 42.3% | 42.4% | 42.2 / 42.5% |

  Patients (slides) per split: train 10 (11), val_es 4 (6), val_cal 6 (7), id_test 6 (6). PathMNIST splits are unchanged
  (checked byte-identical against the committed version); only Camelyon was re-planned and re-downloaded. Candidates:
  `outputs/stage1b/` (`candidates.csv`, `summary.md`).
- **Known limitations (Camelyon, after Phase 1b):**
  - *Remaining prior gap.* Train is 50% tumour while val_es, val_cal, id_test (and OOD) are ~42%; 42% is the most the
    available tumour patches allow when every split needs some. The val_cal vs id_test gap is 0.1 points, so the
    importance-weighted temperature variant is only fitted if that gap reaches 5 points (it does not).
  - *Uneven per-hospital cells.* The overall fractions are equal, but single hospital x split cells are very uneven
    (e.g. id_test hospital 3: 1.9% tumour, val_cal hospital 4: 2.9%, val_es hospital 3: 13.7%), because tumour-rich
    patients are few. Hospital-level analyses inside a split are therefore not balanced.
  - *Few patients per split.* Outside train (10 patients, 11 slides) each source split has only 4-6 patients, so
    in-distribution test/validation estimates are noisy and dominated by patient effects; bootstrap CIs must
    resample patients (the cluster level), not patches.
  - *Spatial overlap.* In WILDS, on average 69% of patches overlap another patch of the same slide (up to 98%; 38% on
    average in the sampled subset (38%); `camelyon_spatial_overlap*.csv`), which justifies grouping above the patch level.
- **Smoke:** uses a synthetic Camelyon-like fixture (random tiles + WILDS-like metadata) that exercises the code only;
  it is never written to `data/`.
## Stage 2 - source classifiers (`python scripts/run_all.py --config configs/full.yaml`, stage 2)
Code: `sfd/train.py` (model, GPU augmentation, safety guard, resumable training), `sfd/calibration.py` (temperature
scaling, NLL/ECE/Brier), `sfd/data/classify_data.py` (per-split uint8 caches, `RowReader`), `sfd/stage2.py`
(orchestration, aggregation, figures, sanity checks), `scripts/stage2_report.py` (per-seed tables).
Done-markers `outputs/stage2/markers/<dataset>_seed<k>.done` are always honoured; per-epoch `last.pt`/`best.pt`
checkpoints (git-ignored) make a stopped run continue at the next epoch.

- **Recipe (both datasets, seeds 0-4):** ResNet-18, ImageNet-pretrained (`IMAGENET1K_V1`), fresh linear head (9 classes
  PathMNIST, 2 classes Camelyon), ImageNet mean/std, SGD lr 0.01 momentum 0.9 weight decay 5e-4, cosine over 8 epochs,
  batch 128, AMP fp16 + channels_last, no label smoothing, early stopping on `val_es` loss (patience 2). Temperature
  fitted on `val_cal` ONLY. The optimizer settings were taken as proposed and not tuned.
- **Augmentation:** random flips and random 90-degree rotations only (per sample, on the GPU). No colour, brightness,
  contrast, blur or noise augmentation, because that would pre-train robustness to the MedMNIST-C corruptions
  (shift family A) and bias the study.
- **Data path:** training batches are read from the SSD cache (`data/classifier_cache/`, git-ignored) with explicit row
  reads (2 ms per 128-image batch). Two things that did not work on this machine: keeping the 3.7 GB PathMNIST train set
  in VRAM next to the desktop (4.7 GB peak of a shared 6 GB card; epoch 3 stalled for 20+ minutes, probably driver
  paging) and `np.memmap` (mapped pages count against Windows "available RAM"; the 2 GB RAM guard stopped the run).
- **Safety guard:** pauses training at slowdown temp - 3 C (99 C), stops if free RAM < 2 GB, logs temperature, free RAM and
  epoch time every epoch (`outputs/stage2/logs/*.jsonl`).
- **Saved per model and split** (git-ignored, `outputs/stage2/preds/`): logits, calibrated probabilities, 512-d
  penultimate features (float16), labels, uid, patient, T. Row order = `splits.parquet` order (checked per file).
  Checkpoints in `outputs/stage2/ckpt/` (git-ignored). Tables and figures that are committed: `metrics_*`,
  `calibration_*`, `seed_stability_*`, `summary_*.md`, `per_seed_results.md/.csv`, `sanity_*.json`, `figures/`.

**Results** (mean ± std over 5 seeds; temperature scaling fitted on `val_cal`; "a -> b" = before -> after scaling;
per-seed values in `outputs/stage2/per_seed_results.md`):

| dataset / split | accuracy | AUROC | NLL | ECE |
|---|---|---|---|---|
| PathMNIST id_test | 0.9956 ± 0.0006 | 1.0000 | 0.0154 -> 0.0154 | 0.0014 -> 0.0015 |
| PathMNIST nct7k_test (family X) | 0.9520 ± 0.0040 | 0.9971 ± 0.0010 | 0.1720 -> 0.1818 | 0.0228 -> 0.0254 |
| Camelyon val_cal | 0.9582 ± 0.0125 | 0.9909 ± 0.0018 | 0.1183 -> 0.1137 | 0.0120 -> 0.0057 |
| Camelyon id_test | 0.9111 ± 0.0442 | 0.9680 ± 0.0105 | 0.3170 -> 0.2712 | 0.0523 -> 0.0411 |
| Camelyon ood_c1 (hospital 1) | 0.8309 ± 0.0341 | 0.9293 ± 0.0319 | 0.5487 -> 0.4797 | 0.1000 -> 0.0832 |
| Camelyon ood_c2 (hospital 2) | 0.6726 ± 0.0396 | 0.8693 ± 0.0080 | 1.5286 -> 1.2459 | 0.2851 -> 0.2727 |

Fitted temperatures: PathMNIST 0.866-0.962 (slightly under-confident), Camelyon 1.136-1.400 (over-confident).
Sanity checks (`sanity_*.json`): no NaN, uid order identical to `splits.parquet` for every split and seed, PathMNIST
ID accuracy 0.9946-0.9961 (threshold 0.95), prediction class counts recorded per seed.

What the numbers say and do not say:
- PathMNIST in-distribution accuracy is at the ceiling and temperature scaling is neutral there (ECE ~0.001-0.002 even
  before). PathMNIST has no patient IDs, so the ID test can share slides with train despite the near-duplicate clustering:
  ID performance is optimistic. On the real centre shift (`nct7k_test`) accuracy drops to 0.952 and scaling with T < 1
  slightly worsens NLL/ECE, as expected when T is fitted in distribution.
- Camelyon shows a clear shift: accuracy 0.911 (ID) -> 0.831 (hospital 1) -> 0.673 (hospital 2), AUROC 0.968 -> 0.929 -> 0.869.
  Temperature scaling lowers NLL and ECE on every Camelyon split, including the shifted ones, but hospital 2 stays badly
  calibrated (ECE 0.27).
- Camelyon seed variance is large (ID accuracy 0.833-0.937). `val_es` has only 4 patients, so early stopping picks
  epoch 2-3 and its loss is noisy; the 6 `id_test` patients make the ID estimate noisy too (see limitations above).
  Report seed spread together with the point estimates.
- The importance-weighted temperature variant was NOT fitted: the val_cal vs id_test label-prior gap is 0.06 points, below
  the 5-point rule (`stage2.calibration.prior_reweight_min_gap_pts`), decided from the computed gap and logged.

## Stage 3 - shift families (`python scripts/run_stage3.py --config configs/full.yaml --steps sample,shortcut,corrupt,inferA,tables`)
Code: `sfd/stage3.py` (steps, tables, figures), `sfd/corrupt.py` (MedMNIST-C wrapper), `sfd/shortcut.py` (cue),
`sfd/download.py` (parallel resumable Range download + MD5). Every step has done-markers; outputs in `outputs/stage3/`.

**Definitions**
- **Sample:** one stratified (by class) sample of 3,000 PathMNIST `id_test` images, seed 0 (`sample_id_test_3000.csv`); the
  SAME images in every family-A/C condition (paired comparisons).
- **Family A (MedMNIST-C v0.1.0):** all 11 corruptions MedMNIST-C defines for PathMNIST (pixelate, jpeg_compression,
  defocus_blur, motion_blur, brightness_up/down, contrast_up/down, saturate, stain_deposit, bubble) at severities 1-5, plus
  clean = 56 conditions, 168,000 images. Source: PathMNIST 224 px from Zenodo 10519652 (`pathmnist_224.npz`, 11.76 GiB,
  MD5 verified), streamed in chunks. Corruptions are applied at 224 px with a per-(corruption, severity, uid) seed, then
  resized to 128 px with **PIL bilinear**: this reproduces the stored 128 px PathMNIST images exactly (RMS 0.0 over 300
  images; bicubic 3.2, cv2-area 2.7, lanczos 4.9, box 6.1 levels; `resize_correspondence.json`). Clean (severity 0) goes
  through the same resize. Determinism: 20 images per condition rebuilt and byte-identical (`corruption_report.json`).
  Inference: the 5 standard PathMNIST models from stage 2, each calibrated with its own stage-2 T.
  Packaging notes: `medmnistc` 0.1.0 hard-pins old `medmnist`/`scikit-image`, so it is installed with `--no-deps`; its
  motion blur needs ImageMagick (7.1.2, installed with the user's approval) + `Wand` 0.7.2 (`--no-deps`) and calls
  `np.fromstring` in binary mode, which NumPy 2 removed: `sfd/corrupt.py` maps it to the exact NumPy-1 equivalent
  (`frombuffer(...).copy()`).
- **Family C (shortcut cue):** a 16x16 class-specific cue (9 colours x 3 textures, `figures/cue_palette.png`) at the top-left
  corner (offset 4 px), pasted after the flips/rot90 augmentation. Training/val_es/val_cal: cue = label with p = 0.95, else a
  random wrong cue (measured 0.9499 aligned in train). 5 shortcut models, same recipe as stage 2. Test states on the 3,000
  sample: aligned, removed (no cue; identical input to the clean image), reversed (cue of class perm[label], a seeded
  derangement saved in `shortcut_design.json`). Family C = reversed; aligned/removed are controls.
- **Families B and X:** reused from stage 2 without recomputation (Camelyon `ood_c1`/`ood_c2`; PathMNIST `nct7k_test`),
  in the same long-format schema, with the ID test sets as references.
- **Provisional silent failure:** wrong AND calibrated confidence >= 0.9 (placeholder; the frozen threshold is Phase 5).

**Outputs:** `predictions_long.parquet` (git-ignored; 1,077,135 rows: family, condition, severity, model_kind, dataset,
seed, uid, patient, label, pred, confidence, correct, silent_failure_provisional), per-file arrays in `preds/` (logits,
calibrated probs, 512-d float16 features, labels, uid, patient, pred, confidence, correct; git-ignored); committed:
`family_summary.csv`, `condition_summary.csv`, `per_corruption_accuracy.csv`, `per_corruption_sf_rate.csv`,
`shortcut_states.csv`, `metrics_per_seed.csv`, `summary_stage3.md`, `sanity_stage3.json`, `figures/`.

**Results** (mean ± std over 5 seeds):

| family / condition | accuracy | mean confidence | ECE | silent-failure rate |
|---|---|---|---|---|
| A clean (3,000 sample) | 0.9958 ± 0.0006 | 0.9961 | 0.0025 | 0.0019 |
| A severity 1 (mean of 11 corruptions) | 0.9207 ± 0.0039 | 0.9698 | 0.0508 | 0.0376 |
| A severity 3 | 0.7739 ± 0.0092 | 0.9306 | 0.1574 | 0.1121 |
| A severity 5 | 0.6242 ± 0.0032 | 0.9007 | 0.2772 | 0.1893 |
| B Camelyon hospital 1 (`ood_c1`) | 0.8309 ± 0.0341 | 0.9130 | 0.0832 | 0.0664 |
| B Camelyon hospital 2 (`ood_c2`) | 0.6726 ± 0.0396 | 0.9453 | 0.2727 | 0.2273 |
| C aligned (shortcut models) | 0.9997 ± 0.0003 | 0.9994 | 0.0005 | 0.0000 |
| C removed | 0.9783 ± 0.0037 | 0.9890 | 0.0121 | 0.0108 |
| C reversed | 0.9543 ± 0.0047 | 0.9749 | 0.0236 | 0.0196 |
| X `nct7k_test` | 0.9520 ± 0.0040 | 0.9763 | 0.0254 | 0.0201 |

Per corruption: brightness_up/down, contrast_down, saturate and the blurs hurt most (accuracy 0.34-0.61 at severity 5,
silent-failure rates 13-36%); bubble, stain_deposit and pixelate barely matter (>= 0.90 at severity 5). Accuracy falls with
severity for every corruption (no rise > 0.5 points; motion_blur is flat over severities 1-3 because MedMNIST-C only grows
the blur radius there). Confidence stays at 0.90-0.97 while accuracy drops, so silent failures grow with severity.

**Limitations / findings to keep in mind**
- **The shortcut does not dominate (family C).** With p = 0.95 and an easy, highly learnable tissue signal (99.6% ID
  accuracy), the shortcut models rely mostly on tissue: removing the cue costs 2.1 points and reversing it 4.5 points; the
  reversed state is NOT the collapse family C was designed to produce (silent-failure rate 2.0%). The cue is used (aligned
  0.9997 > removed 0.9783 > reversed 0.9543 in every seed), but weakly. Decision pending (stronger cue design and retraining).
- Shortcut-model temperatures are 0.55-0.86 (under-confident on cue-bearing val_cal), lower than the standard models (0.87-0.96).
- Seed spread is largest for contrast_down (std 0.07-0.10 at severities 3-5) and brightness_up s3 (0.07): seeds disagree on
  how robust they are to global intensity changes.
- Family A uses `id_test` images, which (PathMNIST has no patient IDs) may share slides with train; corruptions are
  synthetic. Runtimes: shortcut training 22-28 min per seed; corruption generation ~25 min (4 workers); inference ~50 s per
  seed for 56 conditions. Storage: 224 px source 11.76 GiB, corrupted cache 8.1 GB, predictions 1.0 GB, checkpoints 0.6 GB
  (all git-ignored).

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
training batches are read from an SSD cache of raw uint8 arrays with explicit row reads (see Stage 2),
augmentation runs on the GPU,
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
| 1 | Data, frozen splits, leakage audit (+ Phase 1b patient re-assignment for Camelyon) | done |
| 2 | Source classifiers, temperature scaling, saved logits/features (PathMNIST + Camelyon, 5 seeds each) | done |
| 3-9 | Stages 3-9 of the roadmap | not started |

