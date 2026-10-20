# Phase 1b - patient assignment search (seed 0, 30 restarts x 6000 iterations)

OOD tumour fraction (frozen): 0.424. Frozen source-split tumour fractions: {'id_test': 0.31, 'train': 0.5, 'val_cal': 0.5, 'val_es': 0.317} (gap 19.0 pts).

## Top candidates

| mode | rank | score | gap_pts | max_dev_from_ood_pts | train_tumour | val_es_tumour | val_cal_tumour | id_test_tumour | train_n | val_es_n | val_cal_n | id_test_n | train_patients | val_es_patients | val_cal_patients | id_test_patients | train_hospitals | val_es_hospitals | val_cal_hospitals | id_test_hospitals | val_es_min_hosp_share | val_cal_min_hosp_share | id_test_min_hosp_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict_min_hospital_share_10pct | 1 | 0.0008 | 0.1 | 0.0 | 0.5 | 0.424 | 0.423 | 0.424 | 40000 | 3000 | 5000 | 5000 | 10 | 4 | 6 | 6 | 3 | 3 | 3 | 3 | 0.148 | 0.163 | 0.159 |
| strict_min_hospital_share_10pct | 2 | 0.0011 | 0.1 | 0.1 | 0.5 | 0.424 | 0.424 | 0.424 | 40000 | 3000 | 5000 | 5000 | 7 | 9 | 6 | 4 | 3 | 3 | 3 | 3 | 0.236 | 0.169 | 0.138 |
| strict_min_hospital_share_10pct | 3 | 0.0014 | 0.1 | 0.1 | 0.5 | 0.424 | 0.425 | 0.424 | 40000 | 3000 | 5000 | 5000 | 7 | 4 | 9 | 6 | 3 | 3 | 3 | 3 | 0.165 | 0.114 | 0.139 |
| strict_min_hospital_share_10pct | 4 | 0.0014 | 0.1 | 0.1 | 0.5 | 0.423 | 0.424 | 0.424 | 40000 | 3000 | 5000 | 5000 | 7 | 4 | 9 | 6 | 3 | 3 | 3 | 3 | 0.166 | 0.12 | 0.139 |
| strict_min_hospital_share_10pct | 5 | 0.0014 | 0.1 | 0.1 | 0.5 | 0.423 | 0.424 | 0.424 | 40000 | 3000 | 5000 | 5000 | 8 | 4 | 8 | 6 | 3 | 3 | 3 | 3 | 0.166 | 0.124 | 0.139 |
| strict_all_hospitals | 1 | 0.0025 | 0.0 | 0.4 | 0.5 | 0.428 | 0.428 | 0.427 | 40000 | 3000 | 5000 | 5000 | 9 | 4 | 5 | 8 | 3 | 3 | 3 | 3 | 0.056 | 0.095 | 0.203 |
| strict_all_hospitals | 2 | 0.0045 | 0.3 | 0.3 | 0.5 | 0.427 | 0.427 | 0.424 | 40000 | 3000 | 5000 | 5000 | 6 | 9 | 6 | 5 | 3 | 3 | 3 | 3 | 0.219 | 0.196 | 0.158 |
| strict_all_hospitals | 3 | 0.0045 | 0.3 | 0.4 | 0.5 | 0.423 | 0.42 | 0.421 | 40000 | 3000 | 5000 | 5000 | 9 | 7 | 5 | 5 | 3 | 3 | 3 | 3 | 0.138 | 0.203 | 0.236 |
| strict_all_hospitals | 4 | 0.0045 | 0.0 | 0.8 | 0.5 | 0.431 | 0.432 | 0.432 | 40000 | 3000 | 5000 | 5000 | 7 | 8 | 6 | 5 | 3 | 3 | 3 | 3 | 0.13 | 0.051 | 0.144 |
| strict_all_hospitals | 5 | 0.0045 | 0.1 | 0.6 | 0.5 | 0.428 | 0.429 | 0.43 | 40000 | 3000 | 5000 | 5000 | 9 | 6 | 5 | 6 | 3 | 3 | 3 | 3 | 0.154 | 0.185 | 0.219 |
| relaxed_hospitals | 1 | 0.0022 | 0.1 | 0.2 | 0.5 | 0.424 | 0.425 | 0.426 | 40000 | 3000 | 5000 | 5000 | 6 | 9 | 6 | 5 | 3 | 3 | 3 | 3 | 0.118 | 0.205 | 0.045 |
| relaxed_hospitals | 2 | 0.0023 | 0.1 | 0.2 | 0.5 | 0.425 | 0.424 | 0.425 | 40000 | 3000 | 5000 | 5000 | 8 | 7 | 3 | 8 | 3 | 3 | 2 | 3 | 0.242 | 0.0 | 0.246 |
| relaxed_hospitals | 3 | 0.0027 | 0.2 | 0.2 | 0.5 | 0.426 | 0.424 | 0.424 | 40000 | 3000 | 5000 | 5000 | 8 | 7 | 3 | 8 | 3 | 3 | 2 | 3 | 0.241 | 0.0 | 0.187 |
| relaxed_hospitals | 4 | 0.0027 | 0.1 | 0.2 | 0.5 | 0.423 | 0.421 | 0.422 | 40000 | 3000 | 5000 | 5000 | 8 | 7 | 6 | 5 | 3 | 3 | 3 | 3 | 0.065 | 0.213 | 0.047 |
| relaxed_hospitals | 5 | 0.0035 | 0.3 | 0.2 | 0.5 | 0.425 | 0.422 | 0.424 | 40000 | 3000 | 5000 | 5000 | 6 | 5 | 6 | 9 | 2 | 2 | 3 | 3 | 0.0 | 0.048 | 0.095 |



## Best candidate (strict_min_hospital_share_10pct, rank 1)

| split | hospital | patients | slides | patches | tumour |
|---|---|---|---|---|---|
| id_test | 0 | 2 | 2 | 1339 | 0.5 |
| id_test | 3 | 1 | 1 | 793 | 0.019 |
| id_test | 4 | 3 | 3 | 2868 | 0.5 |
| train | 0 | 2 | 2 | 1878 | 0.5 |
| train | 3 | 5 | 5 | 18902 | 0.5 |
| train | 4 | 3 | 4 | 19220 | 0.5 |
| val_cal | 0 | 2 | 3 | 1431 | 0.5 |
| val_cal | 3 | 3 | 3 | 2754 | 0.5 |
| val_cal | 4 | 1 | 1 | 815 | 0.029 |
| val_es | 0 | 1 | 3 | 1569 | 0.5 |
| val_es | 3 | 1 | 1 | 444 | 0.137 |
| val_es | 4 | 2 | 2 | 987 | 0.431 |



Tumour fraction per hospital over the four source splits: {0: 0.5, 3: 0.476, 4: 0.481}

## Download needed for the best candidate

- Same sampler (new random patches): {"patches_total": 69000, "patches_cached": 24748, "patches_new": 44252, "row_groups_to_fetch": 3366, "est_new_download_gib": 7.25}
- Cache-first sampler (reuses already-downloaded patches, identical counts): {"patches_total": 69000, "patches_cached": 58234, "patches_new": 10766, "row_groups_to_fetch": 2727, "est_new_download_gib": 5.86}; split sizes/tumour: {'size': {'id_test': 5000, 'train': 40000, 'val_cal': 5000, 'val_es': 3000}, 'mean': {'id_test': 0.424, 'train': 0.5, 'val_cal': 0.423, 'val_es': 0.424}}
- Estimates use shard size / number of row groups (100 rows each); OOD splits are unchanged (fully cached).
