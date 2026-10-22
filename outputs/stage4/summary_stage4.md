# Stage 4 - descriptive signal results (NOT the Phase 5 tests)

## Frozen tau

| tau | degenerate | accept_frac | val_cal_precision_at_tau | val_cal_acc | model | dataset | kind | seed |
|---|---|---|---|---|---|---|---|---|
| 0.2749754571717238 | accepts every val_cal prediction (val_cal precision 0.9962 >= 0.95) | 1.0 | 0.9961992398479697 | 0.9961992398479697 | pathmnist_standard_seed0 | pathmnist | standard | 0 |
| 0.5006505881594454 | accepts every val_cal prediction (val_cal precision 0.9574 >= 0.95) | 1.0 | 0.9574 | 0.9574 | camelyon17_standard_seed0 | camelyon17 | standard | 0 |
| 0.5655027753777291 | accepts every val_cal prediction (val_cal precision 0.9994 >= 0.95) | 1.0 | 0.9993998799759952 | 0.9993998799759952 | pathmnist_shortcut_seed0 | pathmnist | shortcut | 0 |
| 1.0 | accepts every val_cal prediction (val_cal precision 1.0000 >= 0.95) | 1.0 | 1.0 | 1.0 | pathmnist_shortcutv2_seed0 | pathmnist | shortcutv2 | 0 |
| 0.3375158549343055 | accepts every val_cal prediction (val_cal precision 0.9958 >= 0.95) | 1.0 | 0.9957991598319664 | 0.9957991598319664 | pathmnist_standard_seed1 | pathmnist | standard | 1 |
| 0.5926475742443992 | nan | 0.9734 | 0.9500719128826792 | 0.9394 | camelyon17_standard_seed1 | camelyon17 | standard | 1 |
| 0.5687635750888563 | accepts every val_cal prediction (val_cal precision 0.9994 >= 0.95) | 1.0 | 0.9993998799759952 | 0.9993998799759952 | pathmnist_shortcut_seed1 | pathmnist | shortcut | 1 |
| 1.0 | accepts every val_cal prediction (val_cal precision 1.0000 >= 0.95) | 1.0 | 1.0 | 1.0 | pathmnist_shortcutv2_seed1 | pathmnist | shortcutv2 | 1 |
| 0.3122886176185193 | accepts every val_cal prediction (val_cal precision 0.9958 >= 0.95) | 1.0 | 0.9957991598319664 | 0.9957991598319664 | pathmnist_standard_seed2 | pathmnist | standard | 2 |
| 0.5031416780647917 | accepts every val_cal prediction (val_cal precision 0.9662 >= 0.95) | 1.0 | 0.9662 | 0.9662 | camelyon17_standard_seed2 | camelyon17 | standard | 2 |
| 0.5017131381852654 | accepts every val_cal prediction (val_cal precision 0.9982 >= 0.95) | 1.0 | 0.9981996399279855 | 0.9981996399279855 | pathmnist_shortcut_seed2 | pathmnist | shortcut | 2 |
| 1.0 | accepts every val_cal prediction (val_cal precision 1.0000 >= 0.95) | 1.0 | 1.0 | 1.0 | pathmnist_shortcutv2_seed2 | pathmnist | shortcutv2 | 2 |
| 0.4593029321151433 | accepts every val_cal prediction (val_cal precision 0.9972 >= 0.95) | 1.0 | 0.9971994398879775 | 0.9971994398879775 | pathmnist_standard_seed3 | pathmnist | standard | 3 |
| 0.5020434451902269 | accepts every val_cal prediction (val_cal precision 0.9556 >= 0.95) | 1.0 | 0.9556 | 0.9556 | camelyon17_standard_seed3 | camelyon17 | standard | 3 |
| 0.4911179806892546 | accepts every val_cal prediction (val_cal precision 0.9984 >= 0.95) | 1.0 | 0.9983996799359872 | 0.9983996799359872 | pathmnist_shortcut_seed3 | pathmnist | shortcut | 3 |
| 1.0 | accepts every val_cal prediction (val_cal precision 1.0000 >= 0.95) | 1.0 | 1.0 | 1.0 | pathmnist_shortcutv2_seed3 | pathmnist | shortcutv2 | 3 |
| 0.4094426927609784 | accepts every val_cal prediction (val_cal precision 0.9972 >= 0.95) | 1.0 | 0.9971994398879775 | 0.9971994398879775 | pathmnist_standard_seed4 | pathmnist | standard | 4 |
| 0.5041719739420443 | accepts every val_cal prediction (val_cal precision 0.9722 >= 0.95) | 1.0 | 0.9722 | 0.9722 | camelyon17_standard_seed4 | camelyon17 | standard | 4 |
| 0.5017502008010603 | accepts every val_cal prediction (val_cal precision 0.9988 >= 0.95) | 1.0 | 0.9987997599519904 | 0.9987997599519904 | pathmnist_shortcut_seed4 | pathmnist | shortcut | 4 |
| 1.0 | accepts every val_cal prediction (val_cal precision 1.0000 >= 0.95) | 1.0 | 1.0 | 1.0 | pathmnist_shortcutv2_seed4 | pathmnist | shortcutv2 | 4 |


## Cue mass (Fig 7 input)

| model_kind | condition | cue_mass_l4_mean | cue_mass_l4_std | cue_mass_l3_mean | cue_mass_l3_std |
|---|---|---|---|---|---|
| shortcut | aligned | 0.0186 | 0.0003 | 0.0328 | 0.0059 |
| shortcut | removed | 0.0126 | 0.0002 | 0.024 | 0.0022 |
| shortcut | reversed | 0.0091 | 0.0006 | 0.0132 | 0.003 |
| shortcutv2 | aligned | 0.0338 | 0.0009 | 0.1116 | 0.0112 |
| shortcutv2 | removed | 0.0147 | 0.0003 | 0.0231 | 0.0059 |
| shortcutv2 | reversed | 0.0383 | 0.0003 | 0.1559 | 0.0127 |


## Primary signals: descriptive AUROC (mean over seeds)

| group | dataset | model_kind | signal | n_seeds | error_rate | sf_rate | auroc_error | auroc_error_std | auroc_sf | auroc_sf_std |
|---|---|---|---|---|---|---|---|---|---|---|
| A (all corrupted) | pathmnist | standard | e_js_norm_l4 | 5 | 0.2236 | 0.2207 | 0.6373 | 0.0343 | 0.6331 | 0.0301 |
| A (all corrupted) | pathmnist | standard | r_maha_pred | 5 | 0.2236 | 0.2207 | 0.7682 | 0.0244 | 0.767 | 0.0252 |
| A (all corrupted) | pathmnist | standard | u_1msp | 5 | 0.2236 | 0.2207 | 0.8442 | 0.0091 | 0.8392 | 0.0092 |
| A clean | pathmnist | standard | e_js_norm_l4 | 5 | 0.0042 | 0.0042 | 0.8201 | 0.0377 | 0.8201 | 0.0377 |
| A clean | pathmnist | standard | r_maha_pred | 5 | 0.0042 | 0.0042 | 0.6427 | 0.0403 | 0.6427 | 0.0403 |
| A clean | pathmnist | standard | u_1msp | 5 | 0.0042 | 0.0042 | 0.9787 | 0.0107 | 0.9787 | 0.0107 |
| B (ood_c1+ood_c2) | camelyon17 | standard | e_js_norm_l4 | 5 | 0.2482 | 0.2436 | 0.6576 | 0.0446 | 0.6487 | 0.0387 |
| B (ood_c1+ood_c2) | camelyon17 | standard | r_maha_pred | 5 | 0.2482 | 0.2436 | 0.5773 | 0.0352 | 0.5778 | 0.035 |
| B (ood_c1+ood_c2) | camelyon17 | standard | u_1msp | 5 | 0.2482 | 0.2436 | 0.7363 | 0.007 | 0.7271 | 0.0133 |
| C removed | pathmnist | shortcut | e_js_norm_l4 | 5 | 0.0217 | 0.0207 | 0.5172 | 0.0445 | 0.5105 | 0.0428 |
| C removed | pathmnist | shortcut | r_maha_pred | 5 | 0.0217 | 0.0207 | 0.6201 | 0.0565 | 0.6211 | 0.0549 |
| C removed | pathmnist | shortcut | u_1msp | 5 | 0.0217 | 0.0207 | 0.9562 | 0.012 | 0.9527 | 0.0133 |
| C removed | pathmnist | shortcutv2 | e_js_norm_l4 | 5 | 0.4457 | 0.0601 | 0.5383 | 0.0548 | 0.459 | 0.1118 |
| C removed | pathmnist | shortcutv2 | r_maha_pred | 5 | 0.4457 | 0.0601 | 0.4362 | 0.0882 | 0.2684 | 0.0266 |
| C removed | pathmnist | shortcutv2 | u_1msp | 5 | 0.4457 | 0.0601 | 0.6956 | 0.0909 | 0.1327 | 0.0441 |
| C reversed | pathmnist | shortcut | e_js_norm_l4 | 5 | 0.0457 | 0.0428 | 0.5701 | 0.0992 | 0.55 | 0.1057 |
| C reversed | pathmnist | shortcut | r_maha_pred | 5 | 0.0457 | 0.0428 | 0.4174 | 0.0435 | 0.4048 | 0.0447 |
| C reversed | pathmnist | shortcut | u_1msp | 5 | 0.0457 | 0.0428 | 0.9233 | 0.0125 | 0.9151 | 0.0152 |
| C reversed | pathmnist | shortcutv2 | e_js_norm_l4 | 5 | 0.8996 | 0.7053 | 0.2091 | 0.1133 | 0.1626 | 0.0638 |
| C reversed | pathmnist | shortcutv2 | r_maha_pred | 5 | 0.8996 | 0.7053 | 0.0505 | 0.0598 | 0.0556 | 0.0336 |
| C reversed | pathmnist | shortcutv2 | u_1msp | 5 | 0.8996 | 0.7053 | 0.187 | 0.0528 | 0.0371 | 0.0195 |
| ID | camelyon17 | standard | e_js_norm_l4 | 5 | 0.0889 | 0.0864 | 0.8169 | 0.0246 | 0.8119 | 0.0304 |
| ID | camelyon17 | standard | r_maha_pred | 5 | 0.0889 | 0.0864 | 0.5849 | 0.0721 | 0.5852 | 0.0728 |
| ID | camelyon17 | standard | u_1msp | 5 | 0.0889 | 0.0864 | 0.8343 | 0.0179 | 0.8293 | 0.0196 |
| ID | pathmnist | standard | e_js_norm_l4 | 5 | 0.0044 | 0.0043 | 0.8229 | 0.0198 | 0.8229 | 0.0198 |
| ID | pathmnist | standard | r_maha_pred | 5 | 0.0044 | 0.0043 | 0.7176 | 0.0378 | 0.7173 | 0.0376 |
| ID | pathmnist | standard | u_1msp | 5 | 0.0044 | 0.0043 | 0.9844 | 0.005 | 0.9843 | 0.0049 |
| X | pathmnist | standard | e_js_norm_l4 | 5 | 0.0479 | 0.0477 | 0.631 | 0.0275 | 0.6297 | 0.0289 |
| X | pathmnist | standard | r_maha_pred | 5 | 0.0479 | 0.0477 | 0.7103 | 0.0678 | 0.7096 | 0.0675 |
| X | pathmnist | standard | u_1msp | 5 | 0.0479 | 0.0477 | 0.9317 | 0.0071 | 0.9311 | 0.007 |


## Validity
```
{
 "rows": 1222120,
 "nan_in_signals": false,
 "cam_randomisation": {
  "layer4": {
   "mean_js_orig_vs_randomised": 0.1280510214883271,
   "mean_spearman_orig_vs_randomised": 0.3593211857869569
  },
  "layer3": {
   "mean_js_orig_vs_randomised": 0.4709689365240347,
   "mean_spearman_orig_vs_randomised": 0.0563960630263943
  },
  "model": "pathmnist_standard_seed0",
  "passed": true
 },
 "pred_agreement_with_saved": {
  "standard/ID/id_test": 1.0,
  "standard/A/clean": 1.0,
  "shortcut/C/reversed": 1.0
 },
 "dose_response_non_monotone": [
  [
   "brightness_down",
   "e_js_raw_l3"
  ],
  [
   "brightness_down",
   "e_pca_l3"
  ]
 ]
}
```
