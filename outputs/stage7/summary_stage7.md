# Stage 7 - referral policy (generated from the computed numbers)

Plan: docs/referral_plan_frozen.md. Residual silent-failure rate (sf_090) among accepted cases at an EXACT review budget B; mean over 5 seeds; 95% paired two-level cluster-bootstrap CIs (B=2000). Differences: negative = fewer residual silent failures.

## Held-out A (MedMNIST-C): prevalence of silent failures in the pool = 10.86%

| budget | random | M0 | M1 | M2 | oracle | M1-M0 | M2-M1 | M1-random |
|---|---|---|---|---|---|---|---|---|
| 5% | 10.86% [9.28, 12.10] | 11.43% [9.76, 12.73] | 11.43% [9.76, 12.73] | 11.43% [9.76, 12.73] | 6.17% [4.50, 7.47] | +0.00 pt [-0.00, +0.00] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | +0.57 pt [+0.49, +0.64] increases |
| 10% | 10.86% [9.28, 12.10] | 12.07% [10.31, 13.44] | 11.48% [9.42, 13.01] | 11.48% [9.42, 13.01] | 0.96% [0.04, 2.33] | -0.59 pt [-0.99, -0.32] reduces | +0.00 pt [+0.00, +0.00] no practical difference | +0.62 pt [+0.06, +0.97] increases |
| 20% | 10.86% [9.28, 12.10] | 13.11% [11.24, 14.66] | 8.79% [6.61, 10.69] | 8.79% [6.61, 10.69] | 0.00% [0.00, 0.00] | -4.32 pt [-5.15, -3.43] reduces | +0.00 pt [-0.00, +0.00] no practical difference | -2.07 pt [-3.01, -1.30] reduces |

## Held-out B (Camelyon hospitals): prevalence of silent failures in the pool = 12.12%

| budget | random | M0 | M1 | M2 | oracle | M1-M0 | M2-M1 | M1-random |
|---|---|---|---|---|---|---|---|---|
| 5% | 12.12% [7.91, 16.91] | 12.76% [8.33, 17.80] | 12.76% [8.33, 17.77] | 12.75% [8.33, 17.77] | 7.49% [3.06, 12.53] | -0.00 pt [-0.06, +0.00] no practical difference | -0.00 pt [-0.00, +0.03] no practical difference | +0.64 pt [+0.40, +0.88] increases |
| 10% | 12.12% [7.91, 16.91] | 13.47% [8.79, 18.79] | 13.34% [8.70, 18.49] | 13.39% [8.71, 18.56] | 2.35% [0.00, 7.68] | -0.12 pt [-0.49, +0.00] inconclusive | +0.05 pt [+0.00, +0.15] no practical difference | +1.22 pt [+0.72, +1.72] increases |
| 20% | 12.12% [7.91, 16.91] | 15.09% [9.64, 20.56] | 13.66% [8.79, 18.88] | 13.77% [8.93, 19.04] | 0.00% [0.00, 0.00] | -1.43 pt [-2.47, -0.45] reduces | +0.11 pt [-0.01, +0.42] inconclusive | +1.54 pt [+0.10, +2.64] increases |

## Held-out C (shortcut v2): prevalence of silent failures in the pool = 42.94%

| budget | random | M0 | M1 | M2 | oracle | M1-M0 | M2-M1 | M1-random |
|---|---|---|---|---|---|---|---|---|
| 5% | 42.94% [39.78, 46.19] | 43.91% [40.46, 47.69] | 44.71% [41.38, 48.19] | 44.71% [41.38, 48.19] | 39.93% [36.61, 43.35] | +0.80 pt [+0.31, +1.25] increases | +0.00 pt [+0.00, +0.00] no practical difference | +1.77 pt [+1.32, +2.11] increases |
| 10% | 42.94% [39.78, 46.19] | 43.16% [39.54, 46.89] | 45.37% [41.55, 49.24] | 45.37% [41.55, 49.24] | 36.60% [33.09, 40.21] | +2.21 pt [+1.30, +3.06] increases | +0.00 pt [+0.00, +0.00] no practical difference | +2.43 pt [+0.92, +3.62] increases |
| 20% | 42.94% [39.78, 46.19] | 41.37% [36.41, 45.59] | 44.97% [39.86, 50.01] | 44.97% [39.86, 50.01] | 28.67% [24.73, 32.73] | +3.61 pt [+1.92, +5.27] increases | +0.00 pt [+0.00, +0.00] no practical difference | +2.04 pt [-1.75, +5.12] inconclusive |

## Held-out X (supplement): prevalence of silent failures in the pool = 0.93%

| budget | random | M0 | M1 | M2 | oracle | M1-M0 | M2-M1 | M1-random |
|---|---|---|---|---|---|---|---|---|
| 5% | 0.93% [0.73, 1.14] | 0.70% [0.49, 0.93] | 0.55% [0.41, 0.71] | 0.55% [0.41, 0.71] | 0.00% [0.00, 0.00] | -0.14 pt [-0.29, -0.03] reduces | +0.00 pt [+0.00, +0.00] no practical difference | -0.37 pt [-0.50, -0.25] reduces |
| 10% | 0.93% [0.73, 1.14] | 0.30% [0.20, 0.43] | 0.40% [0.26, 0.55] | 0.40% [0.26, 0.55] | 0.00% [0.00, 0.00] | +0.10 pt [+0.00, +0.20] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | -0.53 pt [-0.69, -0.38] reduces |
| 20% | 0.93% [0.73, 1.14] | 0.14% [0.07, 0.19] | 0.21% [0.12, 0.32] | 0.21% [0.12, 0.32] | 0.00% [0.00, 0.00] | +0.07 pt [-0.01, +0.20] no practical difference | +0.00 pt [+0.00, +0.00] no practical difference | -0.71 pt [-0.90, -0.55] reduces |

## Answer to the primary question (generated)

- held-out A, M1-M0: 5%: +0.00 pt [-0.00, +0.00] (no practical difference); 10%: -0.59 pt [-0.99, -0.32] (reduces); 20%: -4.32 pt [-5.15, -3.43] (reduces)
- held-out A, M2-M1: 5%: +0.00 pt [+0.00, +0.00] (no practical difference); 10%: +0.00 pt [+0.00, +0.00] (no practical difference); 20%: +0.00 pt [-0.00, +0.00] (no practical difference)
- held-out B, M1-M0: 5%: -0.00 pt [-0.06, +0.00] (no practical difference); 10%: -0.12 pt [-0.49, +0.00] (inconclusive); 20%: -1.43 pt [-2.47, -0.45] (reduces)
- held-out B, M2-M1: 5%: -0.00 pt [-0.00, +0.03] (no practical difference); 10%: +0.05 pt [+0.00, +0.15] (no practical difference); 20%: +0.11 pt [-0.01, +0.42] (inconclusive)
- held-out C, M1-M0: 5%: +0.80 pt [+0.31, +1.25] (increases); 10%: +2.21 pt [+1.30, +3.06] (increases); 20%: +3.61 pt [+1.92, +5.27] (increases)
- held-out C, M2-M1: 5%: +0.00 pt [+0.00, +0.00] (no practical difference); 10%: +0.00 pt [+0.00, +0.00] (no practical difference); 20%: +0.00 pt [+0.00, +0.00] (no practical difference)

## Secondary: threshold transfer (realised workload; flagged if |realised - B| > 1 point)

| held-out | method | B | realised workload | flag |
|---|---|---|---|---|
| A | M0 | 5% | 0.0% [0.0, 0.0] | FLAG |
| A | M0 | 10% | 0.0% [0.0, 0.0] | FLAG |
| A | M0 | 20% | 0.7% [0.0, 2.1] | FLAG |
| A | M1 | 5% | 0.1% [0.0, 0.2] | FLAG |
| A | M1 | 10% | 0.4% [0.1, 0.7] | FLAG |
| A | M1 | 20% | 9.0% [5.9, 12.8] | FLAG |
| A | M2 | 5% | 0.1% [0.0, 0.2] | FLAG |
| A | M2 | 10% | 0.4% [0.1, 0.7] | FLAG |
| A | M2 | 20% | 9.0% [5.9, 12.8] | FLAG |
| B | M0 | 5% | 0.0% [0.0, 0.0] | FLAG |
| B | M0 | 10% | 0.0% [0.0, 0.0] | FLAG |
| B | M0 | 20% | 0.0% [0.0, 0.1] | FLAG |
| B | M1 | 5% | 0.0% [0.0, 0.0] | FLAG |
| B | M1 | 10% | 0.0% [0.0, 0.0] | FLAG |
| B | M1 | 20% | 0.0% [0.0, 0.0] | FLAG |
| B | M2 | 5% | 0.0% [0.0, 0.0] | FLAG |
| B | M2 | 10% | 0.0% [0.0, 0.0] | FLAG |
| B | M2 | 20% | 0.0% [0.0, 0.1] | FLAG |
| C | M0 | 5% | 54.2% [52.4, 56.6] | FLAG |
| C | M0 | 10% | 57.2% [55.4, 59.7] | FLAG |
| C | M0 | 20% | 61.4% [60.3, 63.0] | FLAG |
| C | M1 | 5% | 37.1% [32.2, 40.9] | FLAG |
| C | M1 | 10% | 43.3% [40.1, 45.5] | FLAG |
| C | M1 | 20% | 52.0% [49.3, 54.6] | FLAG |
| C | M2 | 5% | 37.1% [32.2, 40.9] | FLAG |
| C | M2 | 10% | 43.3% [40.1, 45.5] | FLAG |
| C | M2 | 20% | 52.0% [49.3, 54.6] | FLAG |
| X | M0 | 5% | 0.0% [0.0, 0.0] | FLAG |
| X | M0 | 10% | 0.0% [0.0, 0.0] | FLAG |
| X | M0 | 20% | 0.9% [0.8, 1.1] | FLAG |
| X | M1 | 5% | 0.0% [0.0, 0.0] | FLAG |
| X | M1 | 10% | 0.1% [0.1, 0.2] | FLAG |
| X | M1 | 20% | 0.9% [0.7, 1.1] | FLAG |
| X | M2 | 5% | 0.0% [0.0, 0.0] | FLAG |
| X | M2 | 10% | 0.1% [0.1, 0.2] | FLAG |
| X | M2 | 20% | 0.9% [0.7, 1.1] | FLAG |

36 of 36 (held-out, method, budget) cells deviate by more than 1 point.
