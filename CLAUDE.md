# Project: Silent failure detection of medical image classifiers under shift

Research question: do representation drift and Grad-CAM explanation drift detect
confident errors (silent failures = wrong prediction with high calibrated confidence)
beyond calibrated predictive uncertainty, and do they lower the residual silent-failure
rate under a fixed human-review budget? Full design: `ROADMAP_Shayan.md`
(section 10 addendum takes precedence over earlier sections; on conflict, ask).

## Fixed decisions (do NOT change without asking)
- Datasets (agreed with group): PathMNIST 128x128 (MedMNIST v2), MedMNIST-C v0.1.0
  (corruptions applied at 224 then resized to 128), Camelyon17-WILDS v1.0 at 96x96.
  Do NOT change dataset versions or resolutions without asking.
- PathMNIST pool = official train+val (100k, near-duplicate-cluster split); official test = family X (`nct7k_test`).
  Camelyon17 is fetched from the HF mirror (never disable TLS verification) or a local archive.
- Sub-dataset (stratified, group-aware by hospital/PATIENT for Camelyon17, fixed seed, saved as manifest)
  is used; its size is a config parameter (decided after the Phase 0 benchmark).
- Model: ResNet-18 initialised from ImageNet (torchvision IMAGENET1K_V1; replaces the earlier from-scratch
  decision: standard in medical imaging, faster convergence, comparable with the group partner's setup),
  5 seeds for core experiments, max 8 epochs with early stopping on source val loss
  (patience 2, see `configs/full.yaml`), temperature scaling on source val.
  PathMNIST core runs use the FULL train set (~90k); val/test stay untouched.
- Families: A = MedMNIST-C severity 1-5; B = Camelyon17 held-out hospitals;
  C = synthetic shortcut cue (reversed at test); X (supplement) = PathMNIST test (CRC-VAL-HE-7K).
- Nested detectors M0 (uncertainty) < M1 (+representation) < M2 (+explanation), fusion
  with positive (monotone) coefficients, evaluated on a HELD-OUT shift family.

## Rules
- All hyperparameters and paths live in `configs/*.yaml`. No hard-coded paths (no `D:\...`).
- Every stage is a script in `scripts/` and resumable (`--resume`); logs go to `outputs/`.
- Never fit references, thresholds, or fusion weights on any shifted evaluation data.
  Add assertions on uid overlap.
- Report every effect with a paired two-level cluster-bootstrap 95% CI (>=2000 resamples);
  secondary families use BH-FDR. Verdict text must be generated from computed numbers.
- Seeds: 5 for core experiments. Set all RNG seeds.
- Windows: no DataLoader workers. Augment on GPU, use `torch.autocast` (fp16) + `channels_last`.
  Batch 128, fall back to 64 + grad accumulation. Keep VRAM < ~5 GB; GPU temp: pause at the card's
  slowdown temp minus 3 C (read from nvidia-smi; 99 C here); stop if free RAM < 2 GB. 6 GB VRAM, 16 GB RAM
  (often mostly used).
- Training data path (Phase 2 lesson): read uint8 batches from the SSD cache with explicit row reads
  (`sfd.data.classify_data.RowReader`, `stage2.train_on_gpu: false`). Do NOT use np.memmap (mapped pages
  shrink Windows "available RAM") and do NOT keep the 3.7 GB PathMNIST train set in VRAM next to the desktop
  (4.7 GB peak of a shared 6 GB card stalled epoch 3 for 20+ min).
- Augmentation: random flips + rot90 ONLY (per sample, on GPU). No colour/brightness/blur/noise: it would
  pre-train robustness to shift family A.
- A frozen `splits.parquet` is read-only for training stages (stage 2 asserts its hash is unchanged).
- Never run tests or heavy jobs while a GPU training job runs (overlapping jobs tripped the 2 GB RAM guard). A CPU job
  next to training must pause below 3 GB free RAM; when killing processes, match the FULL command line.
- Stage 4: references are fitted ONLY on source-train images (whitelist `reference_inputs()`), frozen with a marker
  and checksums before any shifted file is read; signals are normalised on val_cal only. Frozen tau is degenerate for
  19/20 models (val_cal precision >= 95%): use sf_090/sf_095 as the informative silent-failure labels, report all three.
- Stage 5 follows docs/analysis_plan_frozen.md (frozen at e388dd63). Never change the primary label (sf_090), the primary
  signals (Mahalanobis, layer4 normalised CAM JS) or the rotation after seeing results; extra views are sensitivities.
  Report rule-based verdicts together with effect magnitudes (zero-weight fusion can give |delta| ~ 1e-6 with tight CIs).
- Stage 6 follows docs/batch_plan_frozen.md (931fb36) + amendment A1 (4ea78c3). Thresholds set on one source split (val_cal)
  do NOT transfer to another source split (id_test) for window-mean statistics (actual FAR 46/55 cells outside 3-7%): always
  report the actual FAR next to power, compare powers only at matched actual FAR (A1), and never rank statistics at unequal FAR.
- Stage 7 follows docs/referral_plan_frozen.md (aa962a9). Primary referral protocol = EXACT-budget ranking (top B% of the held-out pool
  by score; never transfer thresholds); compare methods only at equal workload against random and an oracle; verdict rule = CI excludes 0
  AND |d| >= 0.1 pt. Silent failures need confidence >= 0.9, so uncertainty-first ranking spends small budgets (< ~19% of the pool) on
  non-silent cases: always report the residual ERROR rate and SF capture next to the residual silent-failure rate. Blueprint-literal
  transferred thresholds fail the +-1 point workload check (36/36); report realised workload with every threshold-based number.
- Stage 8 follows docs/ablation_plan_frozen.md (f0553df). Declared primaries: P1 sf-M1 - sf-M0, P2 sf-M2 - sf-M1, P3 sf-M1 - M1 (residual
  rate at 5/10/20%, unadjusted); the CAM weight wants a NEGATIVE sign (non-negative M2 cannot use it). Gated / targeted fusion is
  EXPLORATORY and post-hoc: report only in its own table ("exploratory, post-hoc"), never in main tables or as a replacement for M0/M1/M2.
- Phase 9: the deliverable is `silent_failure_report.ipynb` (built by `scripts/build_report_notebook.py`, executed with
  `jupyter nbconvert --execute --inplace`, committed WITH outputs). It may read only small committed files (outputs/, docs/); never
  per-row arrays, data or weights. Every number in its text is computed from a saved table; keep exploratory results in their own section.
- Family C primary = v2 (100% cue, 16 px; chosen by the pre-registered pilot on val_es); v1 (95%) is the supplement.
- Stage 3: family A = all 11 MedMNIST-C PathMNIST corruptions, applied at 224 px then PIL-bilinear resize to 128 px
  (exact match to the stored data); one fixed 3,000-image id_test sample for all conditions. `medmnistc` and `Wand` are
  installed with `--no-deps` (hard pins would downgrade packages); ImageMagick 7.1.2 is a system install.
- Run `python scripts/run_all.py --config configs/smoke.yaml` after any change.
- Prefer small, tested functions in `sfd/`. Final deliverable is ONE self-contained
  notebook (built from the modules in the last phase) + requirements.txt + outputs/.
- Never invent, fake, or weaken results or steps.

## Git
- Remote `origin` = https://github.com/shayansalehi1381/silent-failure.git (private), branch `main`.
- Never commit data, weights, secrets, or files > 50 MB; small CSV/PNG/JSON in outputs/ (< 5 MB) ok.
- Conventional Commits; tag each phase `phase-N`; never force-push or rewrite history.

