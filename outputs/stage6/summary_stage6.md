# Stage 6 - batch-level monitoring (plan: docs/batch_plan_frozen.md)

## Actual false-alarm rate on held-out ID windows (size 100; nominal 5%, tolerance 3-7%)

| H | kind | comb | e_cam_l2 | e_cam_l3 | e_cam_l4 | e_img_mean | r_knn_mean | r_maha_mean | r_mmd | u_ks | u_mean | u_p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | standard | 0.563 | 0.647 | 0.303 | 0.066 | 0.059 | 0.161 | 0.11 | 0.26 | 0.354 | 0.37 | 0.384 |
| B | standard | 0.288 | 0.126 | 0.109 | 0.086 | 0.141 | 0.453 | 0.793 | 0.409 | 0.266 | 0.253 | 0.22 |
| C | shortcut | 0.25 | 0.747 | 0.232 | 0.838 | 0.37 | 0.022 | 0.032 | 0.15 | 0.114 | 0.054 | 0.008 |
| C | shortcutv2 | 0.343 | 0.493 | 0.247 | 0.684 | 0.695 | 0.035 | 0.051 | 0.132 | 0.263 | 0.277 | 0.053 |
| X | standard | 0.556 | 0.647 | 0.303 | 0.066 | 0.059 | 0.161 | 0.11 | 0.26 | 0.354 | 0.37 | 0.384 |


Outside tolerance: 46 of 55 (statistic, family) cells.

## Primary contrasts: power on silent windows, held-out family (95% CI)

| H | kind | a | b | power_a | power_b | estimate | verdict | n_windows |
|---|---|---|---|---|---|---|---|---|
| A | standard | comb | u_mean | 0.923 | 0.81 | +0.114 [+0.076, +0.176] | improves | 1655 |
| A | standard | r_mmd | u_mean | 0.34 | 0.81 | -0.469 [-0.570, -0.342] | worse | 1655 |
| A | standard | e_cam_l4 | r_mmd | 0.074 | 0.34 | -0.266 [-0.311, -0.221] | worse | 1655 |
| B | standard | comb | u_mean | 0.371 | 0.432 | -0.061 [-0.232, +0.072] | inconclusive | 4838 |
| B | standard | r_mmd | u_mean | 0.681 | 0.432 | +0.249 [+0.078, +0.379] | improves | 4838 |
| B | standard | e_cam_l4 | r_mmd | 0.118 | 0.681 | -0.563 [-0.739, -0.396] | worse | 4838 |
| C | shortcut | comb | u_mean | 0.937 | 0.937 | +0.000 [-0.023, +0.018] | inconclusive | 142 |
| C | shortcut | r_mmd | u_mean | 0.211 | 0.937 | -0.725 [-0.789, -0.677] | worse | 142 |
| C | shortcut | e_cam_l4 | r_mmd | 0.761 | 0.211 | +0.549 [+0.447, +0.653] | improves | 142 |
| C | shortcutv2 | comb | u_mean | 0.992 | 0.997 | -0.005 [-0.011, +0.000] | inconclusive | 663 |
| C | shortcutv2 | r_mmd | u_mean | 0.11 | 0.997 | -0.887 [-0.908, -0.860] | worse | 663 |
| C | shortcutv2 | e_cam_l4 | r_mmd | 0.937 | 0.11 | +0.827 [+0.780, +0.864] | improves | 663 |
| X | standard | comb | u_mean | 0.943 | 0.921 | +0.022 [-0.001, +0.070] | inconclusive | 368 |
| X | standard | r_mmd | u_mean | 0.516 | 0.921 | -0.405 [-0.437, -0.366] | worse | 368 |
| X | standard | e_cam_l4 | r_mmd | 0.052 | 0.516 | -0.465 [-0.539, -0.389] | worse | 368 |


## Power on silent windows, pooled (95% CI)

| H | kind | size | mask | stat | power | ci_low | ci_high | n_windows |
|---|---|---|---|---|---|---|---|---|
| A | standard | 100 | silent_pooled | u_mean | 0.81 | 0.682 | 0.884 | 1655 |
| A | standard | 100 | silent_pooled | u_p90 | 0.843 | 0.746 | 0.918 | 1655 |
| A | standard | 100 | silent_pooled | u_ks | 0.715 | 0.517 | 0.836 | 1655 |
| A | standard | 100 | silent_pooled | r_mmd | 0.34 | 0.308 | 0.375 | 1655 |
| A | standard | 100 | silent_pooled | e_cam_l4 | 0.074 | 0.054 | 0.091 | 1655 |
| A | standard | 100 | silent_pooled | comb | 0.923 | 0.858 | 0.961 | 1655 |
| B | standard | 100 | silent_pooled | u_mean | 0.432 | 0.201 | 0.751 | 4838 |
| B | standard | 100 | silent_pooled | u_p90 | 0.103 | 0.088 | 0.124 | 4838 |
| B | standard | 100 | silent_pooled | u_ks | 0.388 | 0.232 | 0.65 | 4838 |
| B | standard | 100 | silent_pooled | r_mmd | 0.681 | 0.51 | 0.859 | 4838 |
| B | standard | 100 | silent_pooled | e_cam_l4 | 0.118 | 0.038 | 0.237 | 4838 |
| B | standard | 100 | silent_pooled | comb | 0.371 | 0.218 | 0.597 | 4838 |
| C | shortcut | 100 | silent_pooled | u_mean | 0.937 | 0.9 | 0.967 | 142 |
| C | shortcut | 100 | silent_pooled | u_p90 | 0.88 | 0.822 | 0.915 | 142 |
| C | shortcut | 100 | silent_pooled | u_ks | 0.901 | 0.832 | 0.954 | 142 |
| C | shortcut | 100 | silent_pooled | r_mmd | 0.211 | 0.157 | 0.247 | 142 |
| C | shortcut | 100 | silent_pooled | e_cam_l4 | 0.761 | 0.687 | 0.823 | 142 |
| C | shortcut | 100 | silent_pooled | comb | 0.937 | 0.909 | 0.961 | 142 |
| C | shortcutv2 | 100 | silent_pooled | u_mean | 0.997 | 0.993 | 1.0 | 663 |
| C | shortcutv2 | 100 | silent_pooled | u_p90 | 0.97 | 0.942 | 0.988 | 663 |
| C | shortcutv2 | 100 | silent_pooled | u_ks | 0.775 | 0.629 | 0.874 | 663 |
| C | shortcutv2 | 100 | silent_pooled | r_mmd | 0.11 | 0.091 | 0.133 | 663 |
| C | shortcutv2 | 100 | silent_pooled | e_cam_l4 | 0.937 | 0.893 | 0.973 | 663 |
| C | shortcutv2 | 100 | silent_pooled | comb | 0.992 | 0.985 | 0.999 | 663 |
| X | standard | 100 | silent_pooled | u_mean | 0.921 | 0.873 | 0.957 | 368 |
| X | standard | 100 | silent_pooled | u_p90 | 0.899 | 0.851 | 0.964 | 368 |
| X | standard | 100 | silent_pooled | u_ks | 0.856 | 0.768 | 0.91 | 368 |
| X | standard | 100 | silent_pooled | r_mmd | 0.516 | 0.457 | 0.577 | 368 |
| X | standard | 100 | silent_pooled | e_cam_l4 | 0.052 | 0.038 | 0.069 | 368 |
| X | standard | 100 | silent_pooled | comb | 0.943 | 0.901 | 0.977 | 368 |
