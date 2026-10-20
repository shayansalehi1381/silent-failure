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
- Sub-dataset (stratified, group-aware by hospital/slide, fixed seed, saved as manifest)
  is used; its size is a config parameter (decided after the Phase 0 benchmark).
- Model: ResNet-18 trained FROM SCRATCH (no ImageNet weights, to avoid a pretraining confound),
  5 seeds for core experiments, max 12 epochs with early stopping on source val loss
  (patience 3, see `configs/full.yaml`), temperature scaling on source val.
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
- Windows: no DataLoader workers. Keep uint8 128px data in RAM, augment on GPU, use
  `torch.autocast` (fp16) + `channels_last`. Batch 128, fall back to 64 + grad accumulation.
  Keep VRAM < ~5 GB; warn if GPU temp > ~83 C. 6 GB VRAM, 16 GB RAM (often mostly used).
- Run `python scripts/run_all.py --config configs/smoke.yaml` after any change.
- Prefer small, tested functions in `sfd/`. Final deliverable is ONE self-contained
  notebook (built from the modules in the last phase) + requirements.txt + outputs/.
- Never invent, fake, or weaken results or steps.

## Git
- Remote `origin` = https://github.com/shayansalehi1381/silent-failure.git (private), branch `main`.
- Never commit data, weights, secrets, or files > 50 MB; small CSV/PNG/JSON in outputs/ (< 5 MB) ok.
- Conventional Commits; tag each phase `phase-N`; never force-push or rewrite history.
