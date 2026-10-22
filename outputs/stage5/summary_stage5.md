# Stage 5 - sample-level results (pre-registered plan: docs/analysis_plan_frozen.md)

## Table 2 - held-out family, label sf_090 (point = mean over 5 seeds; paired cluster bootstrap 95% CI)

| heldout | quantity | metric | value | verdict |
|---|---|---|---|---|
| A | M0 | AUROC | 0.6458 [0.6315, 0.6614] |  |
| A | M0 | AUPRC | 0.1354 [0.1159, 0.1476] |  |
| A | M0 | AURC | 0.0550 [0.0476, 0.0628] |  |
| A | M1 | AUROC | 0.7392 [0.7137, 0.7698] |  |
| A | M1 | AUPRC | 0.1768 [0.1641, 0.1890] |  |
| A | M1 | AURC | 0.0402 [0.0307, 0.0492] |  |
| A | M2 | AUROC | 0.7392 [0.7137, 0.7698] |  |
| A | M2 | AUPRC | 0.1768 [0.1641, 0.1890] |  |
| A | M2 | AURC | 0.0402 [0.0307, 0.0492] |  |
| A | M1-M0 | dAUROC | 0.0934 [0.0642, 0.1275] | improves |
| A | M1-M0 | dAUPRC | 0.0414 [0.0274, 0.0582] | improves |
| A | M1-M0 | dAURC | -0.0147 [-0.0185, -0.0100] | improves |
| A | M2-M1 | dAUROC | -0.0000 [-0.0000, -0.0000] | worse |
| A | M2-M1 | dAUPRC | -0.0000 [-0.0000, -0.0000] | worse |
| A | M2-M1 | dAURC | 0.0000 [0.0000, 0.0000] | worse |
| B | M0 | AUROC | 0.5542 [0.5104, 0.6004] |  |
| B | M0 | AUPRC | 0.1220 [0.0806, 0.1717] |  |
| B | M0 | AURC | 0.0813 [0.0501, 0.1200] |  |
| B | M1 | AUROC | 0.5923 [0.5450, 0.6435] |  |
| B | M1 | AUPRC | 0.1328 [0.0885, 0.1889] |  |
| B | M1 | AURC | 0.0740 [0.0448, 0.1097] |  |
| B | M2 | AUROC | 0.5884 [0.5426, 0.6374] |  |
| B | M2 | AUPRC | 0.1312 [0.0876, 0.1850] |  |
| B | M2 | AURC | 0.0748 [0.0454, 0.1111] |  |
| B | M1-M0 | dAUROC | 0.0381 [0.0150, 0.0651] | improves |
| B | M1-M0 | dAUPRC | 0.0108 [0.0036, 0.0215] | improves |
| B | M1-M0 | dAURC | -0.0073 [-0.0138, -0.0025] | improves |
| B | M2-M1 | dAUROC | -0.0038 [-0.0114, 0.0002] | inconclusive |
| B | M2-M1 | dAUPRC | -0.0017 [-0.0052, 0.0000] | inconclusive |
| B | M2-M1 | dAURC | 0.0008 [-0.0000, 0.0026] | inconclusive |
| C | M0 | AUROC | 0.7038 [0.6568, 0.7491] |  |
| C | M0 | AUPRC | 0.5243 [0.4736, 0.5695] |  |
| C | M0 | AURC | 0.2390 [0.2032, 0.2655] |  |
| C | M1 | AUROC | 0.6471 [0.5879, 0.7052] |  |
| C | M1 | AUPRC | 0.4732 [0.4188, 0.5297] |  |
| C | M1 | AURC | 0.2653 [0.2302, 0.2901] |  |
| C | M2 | AUROC | 0.6471 [0.5879, 0.7052] |  |
| C | M2 | AUPRC | 0.4732 [0.4188, 0.5297] |  |
| C | M2 | AURC | 0.2653 [0.2302, 0.2901] |  |
| C | M1-M0 | dAUROC | -0.0567 [-0.0773, -0.0332] | worse |
| C | M1-M0 | dAUPRC | -0.0511 [-0.0688, -0.0354] | worse |
| C | M1-M0 | dAURC | 0.0263 [0.0128, 0.0385] | worse |
| C | M2-M1 | dAUROC | -0.0000 [-0.0000, 0.0000] | inconclusive |
| C | M2-M1 | dAUPRC | -0.0000 [-0.0000, 0.0000] | inconclusive |
| C | M2-M1 | dAURC | 0.0000 [-0.0000, 0.0000] | inconclusive |


## Does M2 improve over M1 out-of-family? (generated from the numbers)

- Held-out A: dAUROC(M2-M1) = -0.0000 [-0.0000, -0.0000] -> **worse**.
- Held-out B: dAUROC(M2-M1) = -0.0038 [-0.0114, 0.0002] -> **inconclusive**.
- Held-out C: dAUROC(M2-M1) = -0.0000 [-0.0000, 0.0000] -> **inconclusive**.
- Held-out A: dAUROC(M1-M0) = 0.0934 [0.0642, 0.1275] -> **improves**.
- Held-out B: dAUROC(M1-M0) = 0.0381 [0.0150, 0.0651] -> **improves**.
- Held-out C: dAUROC(M1-M0) = -0.0567 [-0.0773, -0.0332] -> **worse**.

## Sensitivity (dAUROC, BH q within the sensitivity family)

| variant | heldout | quantity | value | q_bh |
|---|---|---|---|---|
| kNN instead of Mahalanobis | A | M1-M0 | 0.0308 [0.0136, 0.0542] | 0.0 |
| kNN instead of Mahalanobis | A | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.389 |
| kNN instead of Mahalanobis | B | M1-M0 | 0.0160 [0.0038, 0.0340] | 0.0 |
| kNN instead of Mahalanobis | B | M2-M1 | -0.0021 [-0.0064, 0.0002] | 0.3452 |
| kNN instead of Mahalanobis | C | M1-M0 | -0.0388 [-0.0532, -0.0256] | 0.0 |
| kNN instead of Mahalanobis | C | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.212 |
| layer3 CAM | A | M1-M0 | 0.0934 [0.0642, 0.1275] | 0.0 |
| layer3 CAM | A | M2-M1 | -0.0000 [-0.0000, 0.0000] | 0.1527 |
| layer3 CAM | B | M1-M0 | 0.0381 [0.0150, 0.0651] | 0.0 |
| layer3 CAM | B | M2-M1 | -0.0000 [-0.0000, 0.0000] | 0.4404 |
| layer3 CAM | C | M1-M0 | -0.0567 [-0.0773, -0.0332] | 0.0 |
| layer3 CAM | C | M2-M1 | 0.0043 [-0.0059, 0.0212] | 0.4986 |
| layer2 CAM | A | M1-M0 | 0.0934 [0.0642, 0.1275] | 0.0 |
| layer2 CAM | A | M2-M1 | -0.0025 [-0.0079, -0.0000] | 0.0607 |
| layer2 CAM | B | M1-M0 | 0.0381 [0.0150, 0.0651] | 0.0 |
| layer2 CAM | B | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.2337 |
| layer2 CAM | C | M1-M0 | -0.0567 [-0.0773, -0.0332] | 0.0 |
| layer2 CAM | C | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.3828 |
| raw (unnormalised) CAM JS | A | M1-M0 | 0.0934 [0.0642, 0.1275] | 0.0 |
| raw (unnormalised) CAM JS | A | M2-M1 | -0.0082 [-0.0248, 0.0027] | 0.3259 |
| raw (unnormalised) CAM JS | B | M1-M0 | 0.0381 [0.0150, 0.0651] | 0.0 |
| raw (unnormalised) CAM JS | B | M2-M1 | -0.0256 [-0.0505, 0.0021] | 0.1335 |
| raw (unnormalised) CAM JS | C | M1-M0 | -0.0567 [-0.0773, -0.0332] | 0.0 |
| raw (unnormalised) CAM JS | C | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.1638 |
| PCA-embedding CAM distance | A | M1-M0 | 0.0934 [0.0642, 0.1275] | 0.0 |
| PCA-embedding CAM distance | A | M2-M1 | -0.0005 [-0.0030, 0.0006] | 0.3275 |
| PCA-embedding CAM distance | B | M1-M0 | 0.0381 [0.0150, 0.0651] | 0.0 |
| PCA-embedding CAM distance | B | M2-M1 | -0.0160 [-0.0277, 0.0007] | 0.1207 |
| PCA-embedding CAM distance | C | M1-M0 | -0.0567 [-0.0773, -0.0332] | 0.0 |
| PCA-embedding CAM distance | C | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.6443 |
| float 1-MSP/entropy | A | M1-M0 | 0.1180 [0.0840, 0.1528] | 0.0 |
| float 1-MSP/entropy | A | M2-M1 | -0.0201 [-0.0467, -0.0002] | 0.0073 |
| float 1-MSP/entropy | B | M1-M0 | 0.0705 [-0.0282, 0.1913] | 0.2644 |
| float 1-MSP/entropy | B | M2-M1 | -0.0304 [-0.0708, 0.0079] | 0.1921 |
| float 1-MSP/entropy | C | M1-M0 | 0.0581 [0.0303, 0.0907] | 0.0 |
| float 1-MSP/entropy | C | M2-M1 | -0.0000 [-0.0000, 0.0000] | 0.5702 |
| label sf_095 | A | M1-M0 | 0.1088 [0.0735, 0.1534] | 0.0 |
| label sf_095 | A | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.7304 |
| label sf_095 | B | M1-M0 | 0.0339 [0.0128, 0.0607] | 0.0 |
| label sf_095 | B | M2-M1 | -0.0043 [-0.0131, 0.0008] | 0.6144 |
| label sf_095 | C | M1-M0 | -0.0625 [-0.0834, -0.0378] | 0.0 |
| label sf_095 | C | M2-M1 | -0.0000 [-0.0000, -0.0000] | 0.06 |
| label sf_tau | A | M1-M0 | 0.0093 [0.0008, 0.0214] | 0.0369 |
| label sf_tau | A | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.4935 |
| label sf_tau | B | M1-M0 | 0.0112 [0.0000, 0.0272] | 0.212 |
| label sf_tau | B | M2-M1 | 0.0000 [0.0000, 0.0000] | 0.2187 |
| label sf_tau | C | M1-M0 | -0.0005 [-0.0151, 0.0144] | 0.808 |
| label sf_tau | C | M2-M1 | -0.0085 [-0.0223, 0.0025] | 0.3215 |
| label sf_099 | A | M1-M0 | 0.0170 [0.0056, 0.0319] | 0.0027 |
| label sf_099 | A | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.1524 |
| label sf_099 | B | M1-M0 | 0.0167 [0.0000, 0.0404] | 0.212 |
| label sf_099 | B | M2-M1 | 0.0000 [0.0000, 0.0000] | 0.212 |
| label sf_099 | C | M1-M0 | -0.0079 [-0.0254, 0.0112] | 0.4252 |
| label sf_099 | C | M2-M1 | -0.0007 [-0.0027, 0.0000] | 0.6458 |
| v1 weak shortcut for family C | A | M1-M0 | 0.1636 [0.0790, 0.2429] | 0.0 |
| v1 weak shortcut for family C | A | M2-M1 | -0.0000 [-0.0000, 0.0000] | 0.5155 |
| v1 weak shortcut for family C | B | M1-M0 | 0.1165 [0.0690, 0.1722] | 0.0 |
| v1 weak shortcut for family C | B | M2-M1 | 0.0000 [-0.0000, 0.0000] | 0.6023 |
| v1 weak shortcut for family C | C | M1-M0 | -0.2227 [-0.3036, -0.1530] | 0.0 |
| v1 weak shortcut for family C | C | M2-M1 | -0.0000 [-0.0001, 0.0000] | 0.7424 |


## Family C inversion diagnosis
```
{
 "auroc_sf090": {
  "pooled": {
   "n_per_seed": 9000,
   "sf_rate": 0.42935555555555555,
   "acc": 0.5515555555555556,
   "u_log1msp": 0.7037802292303457,
   "r_maha_pred": 0.6007311755331196,
   "r_knn": 0.6385553625530089,
   "e_js_norm_l4": 0.5566953631782546
  },
  "aligned": {
   "n_per_seed": 3000,
   "sf_rate": 0.0,
   "acc": 1.0,
   "u_log1msp": null,
   "r_maha_pred": null,
   "r_knn": null,
   "e_js_norm_l4": null
  },
  "removed": {
   "n_per_seed": 3000,
   "sf_rate": 0.39566666666666667,
   "acc": 0.5542666666666667,
   "u_log1msp": 0.6150228933956544,
   "r_maha_pred": 0.42389768990702814,
   "r_knn": 0.6645504145655118,
   "e_js_norm_l4": 0.5185490662701093
  },
  "reversed": {
   "n_per_seed": 3000,
   "sf_rate": 0.8924,
   "acc": 0.1004,
   "u_log1msp": 0.12248183635495388,
   "r_maha_pred": 0.05210661534286916,
   "r_knn": 0.03925785845764297,
   "e_js_norm_l4": 0.20015276165208534
  }
 },
 "reversed_profile": {
  "correct": {
   "n_per_seed": 301.2,
   "mean_confidence": 0.9826705818827631,
   "class_counts": {
    "0": 15,
    "1": 14,
    "8": 1477
   },
   "median_pct_u_log1msp": 0.9999686998614414,
   "median_pct_r_maha_pred": 0.9993998799759952,
   "median_pct_r_knn": 0.9999611563209415,
   "median_pct_e_js_norm_l4": 0.9687937587517503
  },
  "wrong_confident (sf_090)": {
   "n_per_seed": 2677.2,
   "mean_confidence": 0.9996303309784825,
   "class_counts": {
    "0": 1507,
    "1": 1880,
    "2": 1685,
    "3": 1690,
    "4": 1305,
    "5": 1980,
    "6": 1284,
    "7": 1530,
    "8": 525
   },
   "median_pct_u_log1msp": 0.9993998799759952,
   "median_pct_r_maha_pred": 0.912382476495299,
   "median_pct_r_knn": 0.9657931586317263,
   "median_pct_e_js_norm_l4": 0.8769753950790158
  },
  "wrong_low_conf": {
   "n_per_seed": 21.6,
   "mean_confidence": 0.7175254963740912,
   "class_counts": {
    "0": 3,
    "1": 11,
    "6": 1,
    "8": 93
   },
   "median_pct_u_log1msp": 0.9999798238752431,
   "median_pct_r_maha_pred": 0.9985997199439888,
   "median_pct_r_knn": 0.9999689480951299,
   "median_pct_e_js_norm_l4": 0.9999744393648097
  }
 },
 "reading": "inversion holds WITHIN the reversed state (not a pooling artefact)"
}
```

## Tie report (fraction of tied values per model kind)
```
{
 "camelyon17/standard": {
  "u_log1msp": 0.00015384615384617106,
  "u_1msp": 0.00025384615384616005
 },
 "pathmnist/shortcut": {
  "u_log1msp": 0.0,
  "u_1msp": 0.002885920422887378
 },
 "pathmnist/shortcutv2": {
  "u_log1msp": 0.0,
  "u_1msp": 0.8087434816772626
 },
 "pathmnist/standard": {
  "u_log1msp": 0.01509247686765458,
  "u_1msp": 0.015093527144402596
 }
}
```
