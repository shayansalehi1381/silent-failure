# Stage 8 - ablations and sensitivity (generated from the computed numbers)

Plan: docs/ablation_plan_frozen.md. Endpoint: residual silent-failure rate (sf_090) at exact budgets; differences in percentage points (negative = fewer residual silent failures); 95% paired cluster-bootstrap CIs; P1-P3 are the declared primary contrasts (unadjusted); all other contrasts are secondary (BH within the group, `q_bh`).

## G1: signal groups

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| M1-M0 | A | +0.00 pt [-0.00, +0.00] no practical difference | -0.59 pt [-0.99, -0.32] reduces | -4.32 pt [-5.15, -3.43] reduces |
| M1-M0 | B | -0.00 pt [-0.06, +0.00] no practical difference | -0.12 pt [-0.49, +0.00] inconclusive | -1.43 pt [-2.47, -0.45] reduces |
| M1-M0 | C | +0.80 pt [+0.31, +1.25] increases | +2.21 pt [+1.30, +3.06] increases | +3.61 pt [+1.92, +5.27] increases |
| M0c-M0 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| M0c-M0 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | -0.12 pt [-0.36, +0.00] inconclusive |
| M0c-M0 | C | -0.49 pt [-1.74, +0.00] inconclusive | -0.29 pt [-0.85, +0.00] inconclusive | -0.11 pt [-0.76, +0.77] inconclusive |
| M2-M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| M2-M1 | B | -0.00 pt [-0.00, +0.03] no practical difference | +0.05 pt [+0.00, +0.15] no practical difference | +0.11 pt [-0.01, +0.42] inconclusive |
| M2-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| s_r_maha_pred-s_u_log1msp | A | -1.33 pt [-1.63, -1.00] reduces | -2.76 pt [-3.25, -2.23] reduces | -5.46 pt [-6.23, -4.59] reduces |
| s_r_maha_pred-s_u_log1msp | B | -0.33 pt [-0.95, -0.09] reduces | -0.91 pt [-2.24, -0.31] reduces | -2.97 pt [-5.42, -1.11] reduces |
| s_r_maha_pred-s_u_log1msp | C | +0.88 pt [+0.34, +1.39] increases | +2.52 pt [+1.46, +3.59] increases | +3.79 pt [+2.13, +5.72] increases |
| s_e_js_norm_l4-s_u_log1msp | A | -0.44 pt [-0.62, -0.33] reduces | -1.11 pt [-1.28, -0.96] reduces | -2.44 pt [-2.88, -1.97] reduces |
| s_e_js_norm_l4-s_u_log1msp | B | -0.08 pt [-0.16, -0.02] no practical difference | -0.46 pt [-0.69, -0.26] reduces | -1.98 pt [-2.35, -1.01] reduces |
| s_e_js_norm_l4-s_u_log1msp | C | -1.49 pt [-2.80, -0.45] reduces | -0.23 pt [-2.16, +1.74] inconclusive | +1.80 pt [-1.39, +7.19] inconclusive |
| s_r_knn-s_u_log1msp | A | -0.82 pt [-1.28, -0.57] reduces | -1.98 pt [-2.75, -1.50] reduces | -4.82 pt [-5.93, -3.76] reduces |
| s_r_knn-s_u_log1msp | B | -0.29 pt [-0.54, -0.10] reduces | -0.76 pt [-1.40, -0.32] reduces | -2.49 pt [-3.92, -0.99] reduces |
| s_r_knn-s_u_log1msp | C | -1.33 pt [-2.34, -0.62] reduces | -0.87 pt [-1.80, -0.12] reduces | +0.25 pt [-1.24, +2.42] inconclusive |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'reduces': 23, 'no practical difference': 16, 'inconclusive': 9, 'increases': 6}. 

## G2: fusion constraint (P1 = sf_M1-sf_M0, P2 = sf_M2-sf_M1, P3 = sf_M1-M1)

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| sf_M0-M0 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| sf_M0-M0 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| sf_M0-M0 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| sf_M1-M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| sf_M1-M1 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.01] no practical difference |
| sf_M1-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| sf_M2-M2 | A | -1.25 pt [-1.42, -1.01] reduces | -2.05 pt [-2.46, -1.44] reduces | -1.03 pt [-1.54, -0.43] reduces |
| sf_M2-M2 | B | -0.00 pt [-0.01, +0.00] no practical difference | -0.02 pt [-0.10, +0.00] no practical difference | -0.09 pt [-0.27, +0.01] no practical difference |
| sf_M2-M2 | C | -3.80 pt [-4.43, -3.27] reduces | -5.14 pt [-6.60, -3.14] reduces | -3.92 pt [-6.56, -1.15] reduces |
| sf_M1-sf_M0 | A | +0.00 pt [-0.00, +0.00] no practical difference | -0.59 pt [-0.99, -0.32] reduces | -4.32 pt [-5.15, -3.43] reduces |
| sf_M1-sf_M0 | B | -0.00 pt [-0.06, +0.00] no practical difference | -0.12 pt [-0.49, +0.00] inconclusive | -1.43 pt [-2.47, -0.45] reduces |
| sf_M1-sf_M0 | C | +0.80 pt [+0.31, +1.25] increases | +2.21 pt [+1.30, +3.06] increases | +3.61 pt [+1.92, +5.27] increases |
| sf_M2-sf_M1 | A | -1.25 pt [-1.42, -1.01] reduces | -2.05 pt [-2.46, -1.44] reduces | -1.03 pt [-1.54, -0.43] reduces |
| sf_M2-sf_M1 | B | -0.00 pt [-0.01, +0.02] no practical difference | +0.02 pt [-0.08, +0.13] no practical difference | +0.03 pt [-0.23, +0.38] no practical difference |
| sf_M2-sf_M1 | C | -3.80 pt [-4.43, -3.27] reduces | -5.14 pt [-6.60, -3.14] reduces | -3.92 pt [-6.56, -1.15] reduces |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'no practical difference': 26, 'reduces': 15, 'increases': 3, 'inconclusive': 1}. 

## G3: CAM layer / form

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| M1+e_js_norm_l3-M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| M1+e_js_norm_l3-M1 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| M1+e_js_norm_l3-M1 | C | -0.56 pt [-1.37, +0.00] inconclusive | -0.80 pt [-2.06, +0.00] inconclusive | -0.48 pt [-2.63, +0.82] inconclusive |
| M1+e_js_norm_l2-M1 | A | -0.04 pt [-0.14, +0.00] no practical difference | -0.10 pt [-0.39, +0.00] inconclusive | +0.14 pt [-0.00, +0.42] inconclusive |
| M1+e_js_norm_l2-M1 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| M1+e_js_norm_l2-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| M1+e_js_raw_l4-M1 | A | -0.20 pt [-0.33, -0.08] reduces | -0.49 pt [-0.78, -0.20] reduces | +0.30 pt [-0.12, +0.87] inconclusive |
| M1+e_js_raw_l4-M1 | B | +0.00 pt [+0.00, +0.05] no practical difference | +0.09 pt [+0.00, +0.33] no practical difference | +0.30 pt [-0.26, +0.95] inconclusive |
| M1+e_js_raw_l4-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| M1+e_pca_l4-M1 | A | -0.03 pt [-0.09, +0.00] no practical difference | -0.15 pt [-0.48, +0.00] inconclusive | +0.05 pt [-0.00, +0.17] no practical difference |
| M1+e_pca_l4-M1 | B | +0.00 pt [+0.00, +0.04] no practical difference | +0.08 pt [+0.00, +0.27] no practical difference | +0.38 pt [-0.07, +0.86] inconclusive |
| M1+e_pca_l4-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| sf_M1+e_js_norm_l3-sf_M1 | A | -1.52 pt [-2.12, -1.18] reduces | -2.25 pt [-2.85, -1.46] reduces | -0.52 pt [-1.25, +0.12] inconclusive |
| sf_M1+e_js_norm_l3-sf_M1 | B | -0.21 pt [-0.45, -0.04] reduces | -0.51 pt [-0.92, -0.22] reduces | -0.81 pt [-1.43, +0.02] inconclusive |
| sf_M1+e_js_norm_l3-sf_M1 | C | -2.54 pt [-3.49, -1.56] reduces | -4.27 pt [-6.02, -2.40] reduces | -5.16 pt [-9.09, -1.37] reduces |
| sf_M1+e_js_norm_l2-sf_M1 | A | -0.92 pt [-1.56, -0.31] reduces | -1.55 pt [-2.51, -0.58] reduces | -0.37 pt [-1.19, +0.33] inconclusive |
| sf_M1+e_js_norm_l2-sf_M1 | B | -0.07 pt [-0.18, +0.00] no practical difference | -0.20 pt [-0.47, -0.01] reduces | -0.66 pt [-1.09, -0.10] reduces |
| sf_M1+e_js_norm_l2-sf_M1 | C | -3.06 pt [-3.64, -2.61] reduces | -4.49 pt [-5.55, -3.03] reduces | -5.89 pt [-9.16, -2.09] reduces |
| sf_M1+e_js_raw_l4-sf_M1 | A | -0.39 pt [-0.68, -0.17] reduces | -0.79 pt [-1.18, -0.42] reduces | +0.21 pt [-0.30, +0.84] inconclusive |
| sf_M1+e_js_raw_l4-sf_M1 | B | +0.00 pt [+0.00, +0.05] no practical difference | +0.09 pt [+0.00, +0.33] no practical difference | +0.31 pt [-0.22, +0.95] inconclusive |
| sf_M1+e_js_raw_l4-sf_M1 | C | -2.36 pt [-3.12, -1.43] reduces | -3.11 pt [-5.26, -0.72] reduces | -1.39 pt [-4.59, +1.07] inconclusive |
| sf_M1+e_pca_l4-sf_M1 | A | -1.01 pt [-1.41, -0.56] reduces | -1.90 pt [-2.57, -1.18] reduces | -0.73 pt [-1.56, -0.12] reduces |
| sf_M1+e_pca_l4-sf_M1 | B | +0.00 pt [+0.00, +0.04] no practical difference | +0.08 pt [+0.00, +0.27] no practical difference | +0.40 pt [-0.01, +0.87] inconclusive |
| sf_M1+e_pca_l4-sf_M1 | C | -2.77 pt [-3.47, -2.07] reduces | -3.36 pt [-5.52, -1.18] reduces | -2.17 pt [-5.13, +0.08] inconclusive |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'no practical difference': 30, 'reduces': 25, 'inconclusive': 17}. 

## G4: Mahalanobis vs kNN

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| M1knn-M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.59 pt [+0.32, +0.99] increases | +3.32 pt [+2.82, +3.80] increases |
| M1knn-M1 | B | +0.00 pt [+0.00, +0.06] no practical difference | +0.12 pt [+0.00, +0.49] inconclusive | +1.20 pt [+0.41, +1.93] increases |
| M1knn-M1 | C | -1.98 pt [-2.66, -1.51] reduces | -2.95 pt [-3.68, -2.05] reduces | -3.37 pt [-5.52, -0.85] reduces |
| M2knn-M2 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.59 pt [+0.32, +0.99] increases | +3.32 pt [+2.82, +3.80] increases |
| M2knn-M2 | B | +0.00 pt [+0.00, +0.04] no practical difference | +0.08 pt [+0.00, +0.36] no practical difference | +1.04 pt [+0.38, +1.56] increases |
| M2knn-M2 | C | -1.98 pt [-2.66, -1.51] reduces | -2.95 pt [-3.68, -2.05] reduces | -3.36 pt [-5.52, -0.85] reduces |
| sf_M1knn-sf_M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.59 pt [+0.32, +0.99] increases | +3.32 pt [+2.82, +3.80] increases |
| sf_M1knn-sf_M1 | B | +0.00 pt [+0.00, +0.06] no practical difference | +0.12 pt [+0.00, +0.49] inconclusive | +1.22 pt [+0.46, +1.93] increases |
| sf_M1knn-sf_M1 | C | -1.98 pt [-2.66, -1.51] reduces | -2.95 pt [-3.68, -2.05] reduces | -3.37 pt [-5.52, -0.85] reduces |
| sf_M2knn-sf_M2 | A | +0.12 pt [+0.07, +0.20] increases | +0.27 pt [+0.13, +0.49] increases | +0.52 pt [+0.23, +0.92] increases |
| sf_M2knn-sf_M2 | B | +0.00 pt [+0.00, +0.04] no practical difference | +0.10 pt [+0.00, +0.40] increases | +1.15 pt [+0.46, +1.70] increases |
| sf_M2knn-sf_M2 | C | -0.20 pt [-0.49, +0.11] inconclusive | -0.97 pt [-1.59, -0.20] reduces | -4.21 pt [-7.13, -1.92] reduces |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'increases': 14, 'reduces': 11, 'no practical difference': 8, 'inconclusive': 3}. 

## G5: reference size

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| M1_N500-M1_Nfull | A | -0.01 pt [-0.07, +0.00] no practical difference | -0.33 pt [-0.49, -0.13] reduces | -0.01 pt [-0.20, +0.28] no practical difference |
| M1_N500-M1_Nfull | B | +0.00 pt [+0.00, +0.05] no practical difference | +0.08 pt [+0.00, +0.26] no practical difference | +0.43 pt [+0.13, +0.81] increases |
| M1_N500-M1_Nfull | C | -0.11 pt [-0.28, +0.02] inconclusive | +0.02 pt [-0.48, +0.47] no practical difference | +0.06 pt [-1.09, +0.87] no practical difference |
| M1_N1000-M1_Nfull | A | -0.00 pt [-0.01, +0.00] no practical difference | -0.16 pt [-0.22, -0.05] reduces | +0.08 pt [-0.08, +0.27] no practical difference |
| M1_N1000-M1_Nfull | B | +0.00 pt [+0.00, +0.03] no practical difference | +0.07 pt [+0.00, +0.18] no practical difference | +0.35 pt [+0.13, +0.61] increases |
| M1_N1000-M1_Nfull | C | +0.05 pt [-0.08, +0.12] no practical difference | +0.37 pt [+0.04, +0.67] increases | +0.40 pt [-0.22, +1.17] inconclusive |
| M1_N2000-M1_Nfull | A | +0.00 pt [-0.00, +0.00] no practical difference | -0.11 pt [-0.17, -0.02] reduces | -0.02 pt [-0.12, +0.10] no practical difference |
| M1_N2000-M1_Nfull | B | +0.00 pt [+0.00, +0.02] no practical difference | +0.04 pt [+0.00, +0.12] no practical difference | +0.20 pt [+0.05, +0.38] increases |
| M1_N2000-M1_Nfull | C | -0.01 pt [-0.08, +0.06] no practical difference | +0.07 pt [-0.08, +0.26] no practical difference | -0.14 pt [-0.54, +0.27] inconclusive |
| sf_M1_N500-sf_M1_Nfull | A | -0.01 pt [-0.07, +0.00] no practical difference | -0.33 pt [-0.49, -0.13] reduces | -0.01 pt [-0.20, +0.28] no practical difference |
| sf_M1_N500-sf_M1_Nfull | B | +0.00 pt [+0.00, +0.05] no practical difference | +0.08 pt [+0.00, +0.26] no practical difference | +0.47 pt [+0.18, +0.82] increases |
| sf_M1_N500-sf_M1_Nfull | C | -0.11 pt [-0.27, +0.02] inconclusive | +0.02 pt [-0.48, +0.47] no practical difference | +0.06 pt [-1.09, +0.87] no practical difference |
| sf_M1_N1000-sf_M1_Nfull | A | -0.00 pt [-0.01, +0.00] no practical difference | -0.16 pt [-0.23, -0.05] reduces | +0.08 pt [-0.08, +0.27] no practical difference |
| sf_M1_N1000-sf_M1_Nfull | B | +0.00 pt [+0.00, +0.03] no practical difference | +0.07 pt [+0.00, +0.18] no practical difference | +0.39 pt [+0.17, +0.62] increases |
| sf_M1_N1000-sf_M1_Nfull | C | +0.05 pt [-0.08, +0.12] no practical difference | +0.37 pt [+0.04, +0.67] increases | +0.40 pt [-0.22, +1.17] inconclusive |
| sf_M1_N2000-sf_M1_Nfull | A | +0.00 pt [-0.00, +0.00] no practical difference | -0.11 pt [-0.17, -0.02] reduces | -0.02 pt [-0.12, +0.10] no practical difference |
| sf_M1_N2000-sf_M1_Nfull | B | +0.00 pt [+0.00, +0.02] no practical difference | +0.04 pt [+0.00, +0.12] no practical difference | +0.23 pt [+0.08, +0.38] increases |
| sf_M1_N2000-sf_M1_Nfull | C | -0.01 pt [-0.08, +0.06] no practical difference | +0.07 pt [-0.08, +0.26] no practical difference | -0.14 pt [-0.54, +0.27] inconclusive |
| M1_Nfull-M1 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| M1_Nfull-M1 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |
| M1_Nfull-M1 | C | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'no practical difference': 43, 'increases': 8, 'reduces': 6, 'inconclusive': 6}. 

## G6: temperature scaling (before - after)

| contrast | held-out | 5% | 10% | 20% |
|---|---|---|---|---|
| M0_T1-M0 | A | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | -0.11 pt [-0.18, -0.02] reduces |
| M0_T1-M0 | B | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.00 pt [-0.00, +0.00] no practical difference |
| M0_T1-M0 | C | -0.80 pt [-1.47, -0.30] reduces | -0.26 pt [-1.01, +0.35] inconclusive | +0.45 pt [-0.20, +1.23] inconclusive |
| M1_T1-M1 | A | +0.00 pt [-0.01, +0.00] no practical difference | -0.05 pt [-0.07, -0.04] no practical difference | -0.09 pt [-0.12, -0.06] no practical difference |
| M1_T1-M1 | B | -0.01 pt [-0.03, +0.00] no practical difference | -0.04 pt [-0.14, -0.00] no practical difference | -0.23 pt [-0.35, -0.10] reduces |
| M1_T1-M1 | C | +0.03 pt [-0.04, +0.09] no practical difference | +0.11 pt [-0.08, +0.27] inconclusive | +0.01 pt [-0.06, +0.20] no practical difference |

Verdict counts over these contrasts (3 held-out x 3 budgets): {'no practical difference': 12, 'reduces': 3, 'inconclusive': 3}. 

## Exploratory, post-hoc: targeted fusion (NOT a main-table result)

Gated share (calibrated confidence >= 0.90) of each pool: A 80.7%, B 79.2%, C 97.1%

| held-out | budget | random | M0 | M1 | sf_M1 | G-M0 | G-M1 | G-sf M1 | G-sf M2 | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 5% | 10.86 [9.28, 12.10] | 11.43 [9.76, 12.73] | 11.43 [9.76, 12.73] | 11.43 [9.76, 12.73] | 9.06 [7.54, 10.31] | 8.53 [6.93, 9.90] | 8.53 [6.93, 9.90] | 9.21 [7.53, 10.54] | 6.17 [4.50, 7.47] |
| A | 10% | 10.86 [9.28, 12.10] | 12.07 [10.31, 13.44] | 11.48 [9.42, 13.01] | 11.48 [9.42, 13.01] | 7.37 [6.02, 8.63] | 6.51 [4.87, 7.94] | 6.51 [4.87, 7.94] | 7.40 [5.64, 8.96] | 0.96 [0.04, 2.33] |
| A | 20% | 10.86 [9.28, 12.10] | 13.11 [11.24, 14.66] | 8.79 [6.61, 10.69] | 8.79 [6.61, 10.69] | 4.70 [3.74, 5.67] | 3.54 [2.34, 4.77] | 3.54 [2.34, 4.77] | 3.98 [2.76, 5.32] | 0.00 [0.00, 0.00] |
| B | 5% | 12.12 [7.91, 16.91] | 12.76 [8.33, 17.80] | 12.76 [8.33, 17.77] | 12.76 [8.33, 17.77] | 11.05 [7.04, 15.67] | 10.41 [6.56, 14.92] | 10.41 [6.56, 14.92] | 10.45 [6.57, 14.98] | 7.49 [3.06, 12.53] |
| B | 10% | 12.12 [7.91, 16.91] | 13.47 [8.79, 18.79] | 13.34 [8.70, 18.49] | 13.34 [8.70, 18.49] | 9.92 [6.24, 14.40] | 8.87 [5.40, 13.18] | 8.87 [5.40, 13.18] | 8.92 [5.41, 13.22] | 2.35 [0.00, 7.68] |
| B | 20% | 12.12 [7.91, 16.91] | 15.09 [9.64, 20.56] | 13.66 [8.79, 18.88] | 13.66 [8.79, 18.88] | 7.85 [4.76, 12.16] | 6.67 [3.79, 10.56] | 6.67 [3.79, 10.56] | 6.74 [3.80, 10.65] | 0.00 [0.00, 0.00] |
| C | 5% | 42.94 [39.78, 46.19] | 43.91 [40.46, 47.69] | 44.71 [41.38, 48.19] | 44.71 [41.38, 48.19] | 42.07 [38.80, 45.40] | 44.47 [41.13, 47.95] | 44.47 [41.13, 47.95] | 42.85 [39.79, 46.64] | 39.93 [36.61, 43.35] |
| C | 10% | 42.94 [39.78, 46.19] | 43.16 [39.54, 46.89] | 45.37 [41.55, 49.24] | 45.37 [41.55, 49.24] | 41.30 [37.84, 44.80] | 44.62 [40.52, 48.39] | 44.62 [40.52, 48.39] | 43.38 [40.28, 46.66] | 36.60 [33.09, 40.21] |
| C | 20% | 42.94 [39.78, 46.19] | 41.37 [36.41, 45.59] | 44.97 [39.86, 50.01] | 44.97 [39.86, 50.01] | 39.08 [34.07, 43.24] | 43.26 [38.16, 48.30] | 43.26 [38.16, 48.30] | 42.71 [38.21, 46.27] | 28.67 [24.73, 32.73] |

Targeted contrasts (residual rate, points; exploratory):

- G_M0-M0, held-out A: 5%: -2.37 pt [-2.51, -2.17] reduces; 10%: -4.69 pt [-5.05, -4.16] reduces; 20%: -8.41 pt [-9.20, -7.29] reduces
- G_M0-M0, held-out B: 5%: -1.71 pt [-2.30, -1.16] reduces; 10%: -3.54 pt [-4.77, -2.37] reduces; 20%: -7.24 pt [-9.71, -4.32] reduces
- G_M0-M0, held-out C: 5%: -1.83 pt [-2.69, -1.19] reduces; 10%: -1.86 pt [-2.70, -1.17] reduces; 20%: -2.29 pt [-2.90, -1.61] reduces
- G_M1-M1, held-out A: 5%: -2.90 pt [-3.27, -2.61] reduces; 10%: -4.97 pt [-5.45, -4.34] reduces; 20%: -5.25 pt [-6.04, -4.25] reduces
- G_M1-M1, held-out B: 5%: -2.34 pt [-3.04, -1.61] reduces; 10%: -4.47 pt [-5.81, -3.03] reduces; 20%: -6.99 pt [-9.48, -4.40] reduces
- G_M1-M1, held-out C: 5%: -0.23 pt [-0.52, -0.06] reduces; 10%: -0.75 pt [-1.20, -0.48] reduces; 20%: -1.72 pt [-2.32, -1.04] reduces
- G_sfM1-M1, held-out A: 5%: -2.90 pt [-3.27, -2.61] reduces; 10%: -4.97 pt [-5.45, -4.34] reduces; 20%: -5.25 pt [-6.04, -4.25] reduces
- G_sfM1-M1, held-out B: 5%: -2.34 pt [-3.04, -1.61] reduces; 10%: -4.47 pt [-5.81, -3.03] reduces; 20%: -6.99 pt [-9.48, -4.40] reduces
- G_sfM1-M1, held-out C: 5%: -0.23 pt [-0.52, -0.06] reduces; 10%: -0.75 pt [-1.20, -0.48] reduces; 20%: -1.72 pt [-2.32, -1.04] reduces
- G_sfM1-M0, held-out A: 5%: -2.90 pt [-3.27, -2.61] reduces; 10%: -5.56 pt [-6.21, -5.08] reduces; 20%: -9.57 pt [-10.26, -8.69] reduces
- G_sfM1-M0, held-out B: 5%: -2.34 pt [-3.06, -1.62] reduces; 10%: -4.59 pt [-5.99, -3.13] reduces; 20%: -8.42 pt [-11.19, -5.29] reduces
- G_sfM1-M0, held-out C: 5%: +0.56 pt [-0.07, +1.04] inconclusive; 10%: +1.47 pt [+0.24, +2.51] increases; 20%: +1.89 pt [-0.11, +4.13] inconclusive
- G_sfM2-M1, held-out A: 5%: -2.22 pt [-2.49, -2.01] reduces; 10%: -4.08 pt [-4.55, -3.54] reduces; 20%: -4.81 pt [-5.59, -3.82] reduces
- G_sfM2-M1, held-out B: 5%: -2.31 pt [-3.01, -1.58] reduces; 10%: -4.42 pt [-5.76, -3.01] reduces; 20%: -6.92 pt [-9.41, -4.36] reduces
- G_sfM2-M1, held-out C: 5%: -1.85 pt [-2.99, -0.85] reduces; 10%: -1.99 pt [-3.47, -1.00] reduces; 20%: -2.27 pt [-4.03, -1.07] reduces
- sf_M1-M1, held-out A: 5%: +0.00 pt [+0.00, +0.00] no practical difference; 10%: +0.00 pt [-0.00, +0.00] no practical difference; 20%: +0.00 pt [-0.00, +0.00] no practical difference
- sf_M1-M1, held-out B: 5%: +0.00 pt [+0.00, +0.00] no practical difference; 10%: +0.00 pt [+0.00, +0.00] no practical difference; 20%: +0.00 pt [-0.00, +0.01] no practical difference
- sf_M1-M1, held-out C: 5%: +0.00 pt [+0.00, +0.00] no practical difference; 10%: +0.00 pt [+0.00, +0.00] no practical difference; 20%: +0.00 pt [+0.00, +0.00] no practical difference