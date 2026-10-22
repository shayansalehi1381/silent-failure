# Stage 3 - behaviour under shift (mean Â± std over seeds; provisional silent failure = wrong AND calibrated confidence >= 0.9)

## Per family

| family | condition | severity | dataset | model_kind | acc | conf | ece | sf_rate | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| B | ood_c1 | 0 | camelyon17 | standard | 0.8309 Â± 0.0341 | 0.9130 Â± 0.0112 | 0.0832 Â± 0.0296 | 0.0664 Â± 0.0194 | 5 |
| B | ood_c2 | 0 | camelyon17 | standard | 0.6726 Â± 0.0396 | 0.9453 Â± 0.0098 | 0.2727 Â± 0.0425 | 0.2273 Â± 0.0383 | 5 |
| ID | id_test | 0 | camelyon17 | standard | 0.9111 Â± 0.0442 | 0.9489 Â± 0.0054 | 0.0411 Â± 0.0385 | 0.0392 Â± 0.0231 | 5 |
| A | clean | 0 | pathmnist | standard | 0.9958 Â± 0.0006 | 0.9961 Â± 0.0003 | 0.0025 Â± 0.0007 | 0.0019 Â± 0.0003 | 5 |
| A | all corruptions | 1 | pathmnist | standard | 0.9207 Â± 0.0039 | 0.9698 Â± 0.0026 | 0.0508 Â± 0.0026 | 0.0376 Â± 0.0057 | 5 |
| A | all corruptions | 2 | pathmnist | standard | 0.8575 Â± 0.0083 | 0.9510 Â± 0.0029 | 0.0946 Â± 0.0063 | 0.0692 Â± 0.0072 | 5 |
| A | all corruptions | 3 | pathmnist | standard | 0.7739 Â± 0.0092 | 0.9306 Â± 0.0018 | 0.1574 Â± 0.0090 | 0.1121 Â± 0.0086 | 5 |
| A | all corruptions | 4 | pathmnist | standard | 0.7058 Â± 0.0063 | 0.9133 Â± 0.0017 | 0.2084 Â± 0.0076 | 0.1445 Â± 0.0087 | 5 |
| A | all corruptions | 5 | pathmnist | standard | 0.6242 Â± 0.0032 | 0.9007 Â± 0.0020 | 0.2772 Â± 0.0023 | 0.1893 Â± 0.0046 | 5 |
| C | aligned | 0 | pathmnist | shortcut | 0.9997 Â± 0.0003 | 0.9994 Â± 0.0002 | 0.0005 Â± 0.0002 | 0.0000 Â± 0.0000 | 5 |
| C | aligned | 0 | pathmnist | shortcutv2 | 1.0000 Â± 0.0000 | 1.0000 Â± 0.0000 | 0.0000 Â± 0.0000 | 0.0000 Â± 0.0000 | 5 |
| C | removed | 0 | pathmnist | shortcut | 0.9783 Â± 0.0037 | 0.9890 Â± 0.0034 | 0.0121 Â± 0.0019 | 0.0108 Â± 0.0014 | 5 |
| C | removed | 0 | pathmnist | shortcutv2 | 0.5542 Â± 0.0962 | 0.9760 Â± 0.0081 | 0.4225 Â± 0.0950 | 0.3957 Â± 0.0917 | 5 |
| C | reversed | 0 | pathmnist | shortcut | 0.9543 Â± 0.0047 | 0.9749 Â± 0.0053 | 0.0236 Â± 0.0053 | 0.0196 Â± 0.0033 | 5 |
| C | reversed | 0 | pathmnist | shortcutv2 | 0.1004 Â± 0.0220 | 0.9959 Â± 0.0008 | 0.8962 Â± 0.0212 | 0.8924 Â± 0.0203 | 5 |
| ID | id_test | 0 | pathmnist | standard | 0.9956 Â± 0.0006 | 0.9957 Â± 0.0003 | 0.0015 Â± 0.0003 | 0.0016 Â± 0.0003 | 5 |
| X | nct7k_test | 0 | pathmnist | standard | 0.9520 Â± 0.0040 | 0.9763 Â± 0.0015 | 0.0254 Â± 0.0036 | 0.0201 Â± 0.0029 | 5 |


## Shortcut states (family C = reversed)

| family | condition | severity | dataset | model_kind | acc | conf | ece | sf_rate | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| C | aligned | 0 | pathmnist | shortcut | 0.9997 Â± 0.0003 | 0.9994 Â± 0.0002 | 0.0005 Â± 0.0002 | 0.0000 Â± 0.0000 | 5 |
| C | aligned | 0 | pathmnist | shortcutv2 | 1.0000 Â± 0.0000 | 1.0000 Â± 0.0000 | 0.0000 Â± 0.0000 | 0.0000 Â± 0.0000 | 5 |
| C | removed | 0 | pathmnist | shortcut | 0.9783 Â± 0.0037 | 0.9890 Â± 0.0034 | 0.0121 Â± 0.0019 | 0.0108 Â± 0.0014 | 5 |
| C | removed | 0 | pathmnist | shortcutv2 | 0.5542 Â± 0.0962 | 0.9760 Â± 0.0081 | 0.4225 Â± 0.0950 | 0.3957 Â± 0.0917 | 5 |
| C | reversed | 0 | pathmnist | shortcut | 0.9543 Â± 0.0047 | 0.9749 Â± 0.0053 | 0.0236 Â± 0.0053 | 0.0196 Â± 0.0033 | 5 |
| C | reversed | 0 | pathmnist | shortcutv2 | 0.1004 Â± 0.0220 | 0.9959 Â± 0.0008 | 0.8962 Â± 0.0212 | 0.8924 Â± 0.0203 | 5 |


## Accuracy per corruption and severity (mean over seeds)

| condition | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| brightness_down | nan | 0.8369 | 0.7117 | 0.5791 | 0.4999 | 0.4592 |
| brightness_up | nan | 0.7531 | 0.5763 | 0.4232 | 0.3861 | 0.3414 |
| bubble | nan | 0.9942 | 0.993 | 0.9914 | 0.9892 | 0.9851 |
| clean | 0.9958 | nan | nan | nan | nan | nan |
| contrast_down | nan | 0.9133 | 0.7732 | 0.6055 | 0.5126 | 0.4064 |
| contrast_up | nan | 0.9889 | 0.9231 | 0.79 | 0.6626 | 0.5077 |
| defocus_blur | nan | 0.8432 | 0.7813 | 0.7179 | 0.67 | 0.6138 |
| jpeg_compression | nan | 0.9837 | 0.9591 | 0.8605 | 0.8031 | 0.6359 |
| motion_blur | nan | 0.8445 | 0.8376 | 0.8423 | 0.7271 | 0.5995 |
| pixelate | nan | 0.9942 | 0.991 | 0.9773 | 0.9535 | 0.9045 |
| saturate | nan | 0.9835 | 0.8999 | 0.7517 | 0.6081 | 0.4896 |
| stain_deposit | nan | 0.9928 | 0.9865 | 0.9743 | 0.9514 | 0.9226 |


## Silent-failure rate per corruption and severity

| condition | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| brightness_down | nan | 0.0739 | 0.1345 | 0.2005 | 0.2385 | 0.2516 |
| brightness_up | nan | 0.113 | 0.21 | 0.2965 | 0.3242 | 0.3619 |
| bubble | nan | 0.0024 | 0.0025 | 0.0027 | 0.0035 | 0.004 |
| clean | 0.0019 | nan | nan | nan | nan | nan |
| contrast_down | nan | 0.0347 | 0.1231 | 0.2302 | 0.2853 | 0.3471 |
| contrast_up | nan | 0.0037 | 0.0355 | 0.1037 | 0.17 | 0.2642 |
| defocus_blur | nan | 0.0948 | 0.1124 | 0.1136 | 0.1193 | 0.1331 |
| jpeg_compression | nan | 0.0054 | 0.0183 | 0.0656 | 0.0703 | 0.1775 |
| motion_blur | nan | 0.0753 | 0.0764 | 0.0765 | 0.1297 | 0.1882 |
| pixelate | nan | 0.0021 | 0.0036 | 0.0092 | 0.0189 | 0.0361 |
| saturate | nan | 0.006 | 0.041 | 0.1275 | 0.2163 | 0.294 |
| stain_deposit | nan | 0.0027 | 0.0041 | 0.0071 | 0.0134 | 0.0247 |
