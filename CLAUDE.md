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
- Run `python scripts/run_all.py --config configs/smoke.yaml` after any change.
- Prefer small, tested functions in `sfd/`. Final deliverable is ONE self-contained
  notebook (built from the modules in the last phase) + requirements.txt + outputs/.
- Never invent, fake, or weaken results or steps.

## Git
- Remote `origin` = https://github.com/shayansalehi1381/silent-failure.git (private), branch `main`.
- Never commit data, weights, secrets, or files > 50 MB; small CSV/PNG/JSON in outputs/ (< 5 MB) ok.
- Conventional Commits; tag each phase `phase-N`; never force-push or rewrite history.

