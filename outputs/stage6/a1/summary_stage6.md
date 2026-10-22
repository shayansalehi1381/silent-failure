# Stage 6 - batch-level monitoring (plan: docs/batch_plan_frozen.md)

## Actual false-alarm rate on held-out ID windows (size 100; nominal 5%, tolerance 3-7%)

| H | kind | comb | e_cam_l2 | e_cam_l3 | e_cam_l4 | e_img_mean | r_knn_mean | r_maha_mean | r_mmd | u_ks | u_mean | u_p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | standard | 0.073 | 0.051 | 0.052 | 0.056 | 0.054 | 0.068 | 0.028 | 0.054 | 0.07 | 0.082 | 0.099 |
| B | standard | 0.046 | 0.067 | 0.041 | 0.042 | 0.054 | 0.026 | 0.036 | 0.052 | 0.06 | 0.044 | 0.052 |
| C | shortcut | 0.016 | 0.027 | 0.035 | 0.017 | 0.02 | 0.045 | 0.049 | 0.027 | 0.029 | 0.01 | 0.046 |
| C | shortcutv2 | 0.04 | 0.026 | 0.028 | 0.009 | 0.02 | 0.037 | 0.043 | 0.024 | 0.055 | 0.044 | 0.052 |
| X | standard | 0.079 | 0.051 | 0.052 | 0.056 | 0.054 | 0.068 | 0.028 | 0.054 | 0.07 | 0.082 | 0.099 |


Outside tolerance: 23 of 55 (statistic, family) cells.

## Primary contrasts: power on silent windows, held-out family (95% CI)

| H | kind | a | b | power_a | power_b | estimate | verdict | n_windows |
|---|---|---|---|---|---|---|---|---|
| A | standard | comb | u_mean | 0.383 | 0.518 | -0.135 [-0.229, -0.046] | worse | 1629 |
| A | standard | r_mmd | u_mean | 0.09 | 0.518 | -0.428 [-0.493, -0.380] | worse | 1629 |
| A | standard | e_cam_l4 | r_mmd | 0.063 | 0.09 | -0.027 [-0.064, +0.007] | inconclusive | 1629 |
| B | standard | comb | u_mean | 0.357 | 0.438 | -0.081 [-0.289, +0.081] | inconclusive | 3689 |
| B | standard | r_mmd | u_mean | 0.744 | 0.438 | +0.306 [+0.191, +0.448] | improves | 3689 |
| B | standard | e_cam_l4 | r_mmd | 0.104 | 0.744 | -0.640 [-0.844, -0.408] | worse | 3689 |
| C | shortcut | comb | u_mean | 0.795 | 0.753 | +0.042 [+0.019, +0.059] | improves | 336 |
| C | shortcut | r_mmd | u_mean | 0.036 | 0.753 | -0.717 [-0.792, -0.633] | worse | 336 |
| C | shortcut | e_cam_l4 | r_mmd | 0.193 | 0.036 | +0.158 [+0.115, +0.253] | improves | 336 |
| C | shortcutv2 | comb | u_mean | 0.982 | 0.994 | -0.012 [-0.036, +0.007] | inconclusive | 664 |
| C | shortcutv2 | r_mmd | u_mean | 0.014 | 0.994 | -0.980 [-0.992, -0.967] | worse | 664 |
| C | shortcutv2 | e_cam_l4 | r_mmd | 0.383 | 0.014 | +0.369 [+0.323, +0.414] | improves | 664 |
| X | standard | comb | u_mean | 0.662 | 0.793 | -0.130 [-0.189, -0.093] | worse | 376 |
| X | standard | r_mmd | u_mean | 0.277 | 0.793 | -0.516 [-0.596, -0.395] | worse | 376 |
| X | standard | e_cam_l4 | r_mmd | 0.069 | 0.277 | -0.207 [-0.357, -0.109] | worse | 376 |


## Power on silent windows, pooled (95% CI)

| H | kind | size | mask | stat | power | ci_low | ci_high | n_windows |
|---|---|---|---|---|---|---|---|---|
| A | standard | 100 | silent_pooled | u_mean | 0.518 | 0.467 | 0.587 | 1629 |
| A | standard | 100 | silent_pooled | u_p90 | 0.614 | 0.578 | 0.668 | 1629 |
| A | standard | 100 | silent_pooled | u_ks | 0.393 | 0.358 | 0.443 | 1629 |
| A | standard | 100 | silent_pooled | r_mmd | 0.09 | 0.059 | 0.128 | 1629 |
| A | standard | 100 | silent_pooled | e_cam_l4 | 0.063 | 0.055 | 0.074 | 1629 |
| A | standard | 100 | silent_pooled | comb | 0.383 | 0.304 | 0.431 | 1629 |
| B | standard | 100 | silent_pooled | u_mean | 0.438 | 0.23 | 0.673 | 3689 |
| B | standard | 100 | silent_pooled | u_p90 | 0.065 | 0.033 | 0.1 | 3689 |
| B | standard | 100 | silent_pooled | u_ks | 0.279 | 0.133 | 0.43 | 3689 |
| B | standard | 100 | silent_pooled | r_mmd | 0.744 | 0.541 | 0.887 | 3689 |
| B | standard | 100 | silent_pooled | e_cam_l4 | 0.104 | 0.03 | 0.218 | 3689 |
| B | standard | 100 | silent_pooled | comb | 0.357 | 0.226 | 0.502 | 3689 |
| C | shortcut | 100 | silent_pooled | u_mean | 0.753 | 0.671 | 0.836 | 336 |
| C | shortcut | 100 | silent_pooled | u_p90 | 0.857 | 0.797 | 0.912 | 336 |
| C | shortcut | 100 | silent_pooled | u_ks | 0.634 | 0.518 | 0.802 | 336 |
| C | shortcut | 100 | silent_pooled | r_mmd | 0.036 | 0.027 | 0.049 | 336 |
| C | shortcut | 100 | silent_pooled | e_cam_l4 | 0.193 | 0.152 | 0.296 | 336 |
| C | shortcut | 100 | silent_pooled | comb | 0.795 | 0.697 | 0.874 | 336 |
| C | shortcutv2 | 100 | silent_pooled | u_mean | 0.994 | 0.988 | 0.999 | 664 |
| C | shortcutv2 | 100 | silent_pooled | u_p90 | 0.983 | 0.972 | 0.993 | 664 |
| C | shortcutv2 | 100 | silent_pooled | u_ks | 0.526 | 0.449 | 0.603 | 664 |
| C | shortcutv2 | 100 | silent_pooled | r_mmd | 0.014 | 0.002 | 0.026 | 664 |
| C | shortcutv2 | 100 | silent_pooled | e_cam_l4 | 0.383 | 0.33 | 0.43 | 664 |
| C | shortcutv2 | 100 | silent_pooled | comb | 0.982 | 0.962 | 0.998 | 664 |
| X | standard | 100 | silent_pooled | u_mean | 0.793 | 0.742 | 0.855 | 376 |
| X | standard | 100 | silent_pooled | u_p90 | 0.668 | 0.613 | 0.739 | 376 |
| X | standard | 100 | silent_pooled | u_ks | 0.665 | 0.612 | 0.724 | 376 |
| X | standard | 100 | silent_pooled | r_mmd | 0.277 | 0.182 | 0.423 | 376 |
| X | standard | 100 | silent_pooled | e_cam_l4 | 0.069 | 0.041 | 0.088 | 376 |
| X | standard | 100 | silent_pooled | comb | 0.662 | 0.625 | 0.718 | 376 |
