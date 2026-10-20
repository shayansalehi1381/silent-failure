# رودمپ و پایپلاین پروژه — Silent Failure در طبقه‌بندهای تصویر پزشکی (نسخه‌ی شایان)

> تاریخ تنظیم: ۲۰ اکتبر ۲۰۲۶ (۲۸ مهر ۱۴۰۵) — ددلاین تغییرات نهایی: ۵ آبان (حدود ۲۷ اکتبر) → **۷ روز**
> هدف: یک نوت‌بوک/ریپوی **کامل‌تر، تمیزتر و قابل‌دفاع‌تر** از نسخه‌ی آیدا، چون در نهایت یکی از دو پروژه انتخاب می‌شود.

---

## ۰. وضعیت فعلی (بر اساس نوت‌بوک آیدا)

### آیدا چه کرده
- ۱۰ مرحله (۰ تا ۹) روی PathMNIST، Camelyon17-WILDS و شیفت مصنوعی shortcut؛ ResNet-18 با ۵ seed؛ temperature scaling؛ سیگنال‌های uncertainty / Mahalanobis / kNN / Grad-CAM؛ مدل‌های تو در توی M0/M1/M2؛ سطح نمونه + سطح batch + سیاست referral؛ bootstrap خوشه‌ای، BH-FDR، ablation.
- همه‌ی مرحله‌ها اجرا شده‌اند. کیفیت آماری بالاست.

### نقاط ضعف قابل‌بهره‌برداری (چیزی که ما باید بهتر کنیم)
| # | ضعف نسخه‌ی آیدا | فرصت ما |
|---|---|---|
| ۱ | اجرا با `PROFILE="lite"` روی **CPU**: داده‌ی زیرنمونه‌گیری‌شده، فقط ۳ epoch (PathMNIST) و ۲ epoch (Camelyon). خود نوت‌بوک می‌گوید «مطالعه‌ی مقیاس کوچک». تنها آیتم false چک‌لیست Q1 همین است. | اجرای **کامل روی GPU** با داده‌ی کامل و epoch کافی |
| ۲ | نتیجه‌ی اصلی منفی است (explanation drift چیزی اضافه نمی‌کند). نتیجه‌ی منفی ممکن است ناشی از مدل کم‌آموزش‌دیده و روش ضعیف باشد، نه ماهیت مسئله. | روش‌های explanation قوی‌تر + مدل کاملاً آموزش‌دیده → نتیجه‌ی محکم‌تر (چه مثبت چه منفی) |
| ۳ | فقط یک معماری (ResNet-18)، فقط یک روش Grad-CAM در یک لایه | دومین معماری + چند روش explanation |
| ۴ | baselineهای uncertainty محدود (MSP/entropy) | افزودن energy، ensemble، MC-dropout و ... تا مقایسه منصفانه‌تر باشد |
| ۵ | یک نوت‌بوک ~۳۱۲ سلولی با کتابخانه‌ی embedded و مسیرهای ویندوز (`D:\...`) | ریپوی ماژولار + config + اجرای یک‌دستوری + نوت‌بوک تمیز برای ارائه |
| ۶ | هر مرحله ساعت‌ها طول کشیده (روی CPU) | pipeline چک‌پوینت‌دار و سریع روی GPU |

### نقاط قوت او که باید حداقل برابرشان کنیم
leakage audit، held-out shift family، cluster bootstrap، BH-FDR، sensitivity analyses، جدول tidy. **این‌ها را حذف نکنیم؛ همه باید در کار ما هم باشند.**

---

## ۱. استراتژی برد (۵ اصل)

1. **مقیاس واقعی:** مدل‌ها روی GPU با داده‌ی کامل و epoch کافی. (طبق قرارِ گروه: PathMNIST **۱۲۸×۱۲۸**، MedMNIST-C v0.1.0 با خرابی روی ۲۲۴ و resize به ۱۲۸، Camelyon17-WILDS v1.0 در **۹۶×۹۶**.)
2. **نوآوری روش‌شناختی کوچک ولی واقعی:** فقط تکرار پایپلاین کافی نیست. یک روش explanation-drift بهتر + baselineهای قوی‌تر.
3. **صداقت آماری:** هر ادعا با CI و آزمون چندگانه؛ نتیجه‌ی منفی هم اگر منفی بود، **با قدرت آماری کافی** گزارش شود.
4. **مهندسی تمیز:** هر کسی (استاد/داور) بتواند با یک دستور همه چیز را بازتولید کند.
5. **روایت روشن:** خروجی نهایی = ۳ جدول + ۷ شکل + یک پاراگراف نتیجه، نه فقط کد.

> **مهم:** هسته‌ی کار باید با دیتاست‌ها/نسخه‌هایی که با گروه هماهنگ شده بماند. هر چیز اضافه (معماری دوم، دیتاست اضافه) فقط در بخش **supplementary** باشد تا با قرار گروه تداخل نکند.

---

## ۲. پیش‌نیازها (روز ۰ — امروز)

### سؤال‌هایی که باید قبل از شروع روشن شوند
- [ ] محتوای دو ویس ابولفضل چیست و «۲ نتیجه‌ی مشخص برای هر نفر» یعنی چه؟ (روی انتخاب Stage بعدی اثر دارد)
- [ ] آیا ابولفضل زیرمجموعه‌ی مشخصی (sub-dataset) تعیین کرده؟ اگر بله، همان را به‌عنوان اجرای اصلی بگیریم.
- [ ] نمایش/ارائه‌ی نهایی چه فرمتی دارد؟ (نوت‌بوک؟ گزارش؟ ریپو؟)
- [ ] معیار انتخاب بین دو پروژه چیست؟ (دقت علمی؟ تمیزی کد؟ نوآوری؟)

### سخت‌افزار (یکی را انتخاب کن)
| گزینه | نکته |
|---|---|
| GPU شخصی ≥ 8GB | بهترین. |
| Kaggle (T4×2، سقف زمان هفتگی) | رایگان، session محدود → **حتماً resumable** بنویس. |
| Colab (T4) | session قطع می‌شود → ذخیره‌ی چک‌پوینت روی Drive. |
| اجاره‌ی GPU ساعتی | اگر زمان کم آمد. |

### محیط
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision medmnist wilds scikit-learn scipy pandas pyarrow \
    statsmodels matplotlib seaborn tqdm pyyaml pytorch-grad-cam
```
> قبل از هر چیز نسخه‌ی کتابخانه‌ها را در `requirements.txt` قفل کن و `seed` را همه‌جا ثابت کن.

### ساختار ریپو
```
silent-failure/
├── CLAUDE.md                 # قوانین پروژه برای Claude Code (بخش ۱۰)
├── README.md
├── configs/
│   ├── smoke.yaml            # اجرای سریع ۵ دقیقه‌ای برای تست
│   ├── full.yaml             # اجرای اصلی
│   └── ablation.yaml
├── sfd/
│   ├── data/                 # لودر، splitها، leakage audit
│   ├── models/               # ResNet-18، (اختیاری) DenseNet/ViT-S
│   ├── train.py              # آموزش + temperature scaling
│   ├── shifts.py             # A, B, C
│   ├── signals/
│   │   ├── uncertainty.py    # MSP, entropy, energy, ensemble
│   │   ├── representation.py # Mahalanobis (class-cond.), kNN, MMD
│   │   └── explanation.py    # Grad-CAM, Grad-CAM++, (IG)، JS drift
│   ├── detect.py             # M0/M1/M2، fusion held-out-family
│   ├── batch.py              # سطح batch
│   ├── referral.py           # accept/review/refer
│   ├── stats.py              # bootstrap خوشه‌ای، BH-FDR
│   └── viz.py
├── scripts/run_stage{0..9}.py
├── outputs/                  # git-ignored، فقط جدول‌ها و شکل‌ها commit شوند
├── tests/                    # تست‌های واحد کوچک (leakage، threshold، bootstrap)
└── notebook/SilentFailure_final.ipynb   # فقط نمایش نتایج (کد سنگین در sfd/)
```

---

## ۳. مرحله‌ها (مرحله‌به‌مرحله)

هر مرحله: **هدف ← کارها ← خروجی ← معیار قبولی (exit criterion)**.

### Stage 0 — بازتولیدپذیری (۰.۵ روز)
- **کارها:** config واحد (YAML)، ثبت نسخه‌ی کتابخانه‌ها/GPU/seed، `run_stage.py` با قابلیت `--resume`، لاگ.
- **خروجی:** `env_record.json`، اجرای `smoke.yaml` موفق از ابتدا تا انتها.
- **قبولی:** smoke کل پایپلاین را در چند دقیقه می‌رود (حتی با نتایج بی‌معنی).

### Stage 1 — داده، split ثابت، leakage audit (۰.۵–۱ روز)
- **کارها:**
  1. دانلود PathMNIST (۱۲۸px) و Camelyon17-WILDS v1.0 (۹۶px) و MedMNIST-C v0.1.0.
  2. split ثابت و ذخیره‌ی آن (`splits.parquet` با uid، گروه/بیمار، split).
  3. leakage audit: تکرار دقیق + near-duplicate (hash ادراکی/فاصله‌ی ویژگی)، بررسی هم‌پوشانی بیمار/مرکز بین train و test.
  4. گروه‌بندی برای Camelyon: بیمار/slide.
  5. checksum فایل‌ها.
- **خروجی:** Table 1 (دیتاست، دامنه، split، تعداد، مجوز)، `dedup_report.csv`.
- **قبولی:** assert خودکار «هیچ uid بین fitting و deployed مشترک نیست».
- **توجه:** PathMNIST: train/val از NCT-CRC-HE-100K و test رسمی از CRC-VAL-HE-7K (مرکز بالینی متفاوت) → این خودش یک شیفت واقعی است و می‌شود family X.

### Stage 2 — طبقه‌بند مبدأ و کالیبراسیون (۱ روز)
- **کارها:** ResNet-18 (ImageNet-pretrained یا scratch؛ **یکی را انتخاب و مستند کن**)، ۵ seed، early stopping روی val مبدأ، **epoch کافی** (نه ۲–۳)، temperature scaling روی val_cal.
- **خروجی:** logit/feature هر split، `temperatures.csv`، ECE/NLL/Brier قبل و بعد، reliability diagram.
- **قبولی:** accuracy درون‌توزیعی پایدار بین seedها (sd کوچک).
- **اضافه (برای برتری):** معماری دوم (مثلاً DenseNet-121 یا ViT-Small) فقط روی ۱–۲ seed به‌عنوان supplement.

### Stage 3 — تولید شیفت‌ها (۱ روز)
سه خانواده + یک مکمل:
- **A:** MedMNIST-C شدت ۱ تا ۵ (نوع‌های خرابی متنوع).
- **B:** Camelyon17 با بیمارستان‌های held-out.
- **C:** shortcut مصنوعی: نشانه‌ای که در train با کلاس هم‌بسته است و در test برعکس می‌شود. مدل shortcut جدا آموزش بده (۵ seed).
- **X (supplement):** PathMNIST آموزش→CRC-VAL-HE-7K.
- **خروجی:** شکل نمونه‌ی شیفت‌ها (Figure 2)، جدول tidy پیش‌بینی هر تصویر (`model × seed × image × condition`).
- **قبولی:** افت دقت با شدت شیفت منطقی است (monotone تقریبی)، میزان تکیه بر shortcut اندازه‌گیری شده.

### Stage 4 — مرجع‌های مانیتورینگ و سیگنال‌ها (۱.۵ روز) ← **قلب نوآوری**
سیگنال‌ها (هر کدام یک ستون در جدول per-image):
1. **Uncertainty (baseline قوی):** MSP، entropy، **energy score**، (اختیاری) deep ensemble روی ۵ seed موجود (ارزان!)، MC-dropout.
2. **Representation:** Mahalanobis **class-conditional**، kNN، (batch) MMD.
3. **Explanation (جای نوآوری):**
   - Grad-CAM مرجع (لایه‌ی layer3 + حساسیت به layer2/layer4)، مقایسه‌ی class-conditional با مرجع.
   - **Grad-CAM++ / ScoreCAM / Integrated Gradients** به‌عنوان جایگزین.
   - normalize کردن با تغییرپذیری پایه (flip/rot/brightness) تا drift واقعی از نویز جدا شود.
   - معیارهای عددی: JS divergence، آنتروپی نقشه، تمرکز جرم (center-of-mass)، فاصله در فضای PCA.
4. **ایده‌ی پیشنهادی ما:** «feature attribution **تأییدشده با ابزار شناخته‌شده**» — یعنی ابتدا در family C (که cue را می‌شناسیم) بسنجیم هر روش explanation واقعاً cue را می‌بیند یا نه (validity check)، و فقط روش‌های معتبر را وارد fusion کنیم. این جمله‌ی داور را پوشش می‌دهد: «آیا explanation اصلاً معنادار است؟».
- **خروجی:** `monitoring_table.parquet` (percentile هر سیگنال نسبت به مرجع مبدأ)، Figure 7 (case study).
- **قبولی:** مرجع‌ها فقط از train/val_cal مبدأ ساخته شده‌اند؛ هیچ shifted image‌ای در fit نیست (assert).

### Stage 5 — تشخیص silent failure در سطح نمونه (۱ روز) ← **آزمون اصلی نوآوری**
- **تعریف label:** silent failure = پیش‌بینی غلط + confidence بالاتر از آستانه‌ی ثابت (کوچک‌ترین آستانه با precision ≥ ۹۵٪ روی val_cal مبدأ).
- **مدل‌های تو در تو:** M0 (uncertainty) ⊂ M1 (+representation) ⊂ M2 (+explanation) — fusion با ضریب‌های **مثبت (monotone)**.
- **Held-out family:** fusion روی دو خانواده فیت و روی سومی ارزیابی شود (A,B,C هر سه نوبت).
- **متریک‌ها:** AUROC، AUPRC، AURC؛ CI از bootstrap دو‌سطحی (image درون cluster) با ≥۲۰۰۰ تکرار؛ تفاوت‌های جفتی M1−M0 و M2−M1.
- **اضافه (برای برتری):**
  - تحلیل «in-family vs. out-of-family» (آیا explanation اطلاعات دارد ولی انتقال نمی‌یابد؟).
  - jackknife برای تعداد خوشه‌ی کم.
  - مقایسه با baselineهای قوی‌تر Stage 4.
- **خروجی:** Table 2، Figures 3–4، جدول `primary_contrasts.csv`.
- **قبولی:** تمام contrastها با CI گزارش شوند؛ نتیجه (مثبت/منفی/نامشخص) از روی عدد خودکار تولید شود، نه دستی.

### Stage 6 — مانیتورینگ سطح batch (۱ روز)
- **کارها:** پنجره‌های ۱۰۰ تایی با سهم shift ∈ {۰.۱, ۰.۲۵, ۰.۵, ۰.۷۵, ۱}؛ آستانه‌ی هشدار از null منبع (resample) با false-alarm 5٪؛ آماره‌ها: میانگین uncertainty، MMD، JS توضیح، ترکیب‌ها.
- **خروجی:** power و تأخیر تشخیص (detection delay)، Figure 5، بخش batch از Table 3.
- **قبولی:** false-alarm واقعی روی پنجره‌های مبدأ نزدیک ۵٪ (مثلاً ۳–۷٪) — در غیر این صورت کالیبراسیون را اصلاح کن.
- **نکته:** در نسخه‌ی آیدا false-alarm روی family B برای بعضی مانیتورها بالا بود؛ اگر ما درست کنیم، امتیاز است.

### Stage 7 — سیاست accept/review/refer (۰.۵–۱ روز)
- بودجه‌ی بازبینی ۵٪، ۱۰٪، ۲۰٪؛ آستانه‌ها روی دو خانواده‌ی tuning و اعمال روی خانواده‌ی held-out.
- **متریک اصلی:** نرخ silent failure در موارد accept‌شده (residual)، مقایسه با random، M0، M1، M2.
- **خروجی:** Figure 6، بخش referral از Table 3.
- **قبولی:** workload واقعی نزدیک بودجه است (گزارش drift بار کاری).

### Stage 8 — استحکام و ablation (۱ روز)
- اثر severity، اثر نوع خرابی، تفکیک shortcut (cue present/removed/reversed)، raw vs. temperature-scaled، حساسیت Grad-CAM (لایه/PCA/رزولوشن)، حذف patchهای artefact در Camelyon، پایداری بین seed، BH-FDR روی خانواده‌ی ثانویه.
- **Figure 1:** شمای کلی مطالعه.

### Stage 9 — جمع‌بندی (۱ روز)
- جدول evidence map، چک‌لیست «Q1-ready»، نقشه‌ی تطبیق با Blueprint، پاراگراف نتیجه‌ی خودکار.
- **نوت‌بوک نهایی:** فقط نمایش جدول/شکل + توضیح. کد سنگین در `sfd/`.
- README: دستور اجرا، زمان تقریبی، نیاز سخت‌افزاری، ساختار خروجی.

---

## ۴. برنامه‌ی ۷ روزه

| روز | تاریخ | کار | خروجی |
|---|---|---|---|
| ۰ | ۲۰ اکتبر | سؤال‌های بخش ۲، تنظیم ریپو، `CLAUDE.md`، دانلود دیتا شروع شود | ریپو + smoke اولیه |
| ۱ | ۲۱ | Stage 0–1 کامل، شروع آموزش | splits + Table 1 |
| ۲ | ۲۲ | Stage 2 (آموزش ۵ seed) — در پس‌زمینه؛ همزمان کد Stage 3 | مدل‌ها + کالیبراسیون |
| ۳ | ۲۳ | Stage 3 + آموزش مدل‌های shortcut؛ شروع Stage 4 | شیفت‌ها |
| ۴ | ۲۴ | Stage 4 (سیگنال‌ها) + Stage 5 | جدول مانیتورینگ + Table 2 |
| ۵ | ۲۵ | Stage 6 + 7 | Figures 5–6 |
| ۶ | ۲۶ | Stage 8 + 9 | همه‌ی جدول/شکل‌ها |
| ۷ | ۲۷ | تمیزکاری، README، نوت‌بوک نهایی، تست از صفر روی smoke | تحویل |

> **ریسک زمان:** مرحله‌های آموزش و Grad-CAM سنگین‌اند. هنگام روز ۱ با اجرای `smoke` زمان واقعی هر مرحله را روی GPU خودت **اندازه بگیر** و برنامه را اصلاح کن. اگر زمان کم آمد، اولویت حذف: معماری دوم → deep ensemble → ScoreCAM/IG → X supplement. **هرگز حذف نشوند:** leakage audit، held-out family، CI و FDR.

---

## ۵. ریسک‌ها و راه‌حل

| ریسک | راه‌حل |
|---|---|
| قطع شدن session کولب/کگل | چک‌پوینت هر epoch + `--resume`؛ خروجی روی Drive |
| دانلود Camelyon17 سنگین (ده‌ها GB) | شروع دانلود از روز ۰؛ در صورت لزوم sub-dataset طبق نظر ابولفضل |
| Grad-CAM برای همه‌ی تصاویر کند است | batch کردن، ذخیره‌ی نقشه‌ی ۸×۸ (grid) نه تصویر کامل |
| نتیجه‌ی explanation دوباره منفی شد | اشکالی ندارد؛ با CI، validity check و تحلیل in-family/out-of-family به‌عنوان یافته‌ی محکم منفی گزارش شود |
| تداخل با کار آیدا | هر دو نسخه‌ی دیتا/نسخه را مطابق قرار گروه نگه دار؛ مقایسه‌ی متقابل مجاز است |
| نشت داده | assertهای خودکار در Stage 1 و 4 |

---

## ۶. معیار «بهتر از آیدا» (چک‌لیست نهایی)

- [ ] اجرای **full** روی GPU (داده‌ی کامل، epoch کافی) ← مهم‌ترین تمایز
- [ ] baseline uncertainty قوی‌تر (energy / ensemble)
- [ ] ≥ ۲ روش explanation + validity check روی family C
- [ ] (supplement) معماری دوم
- [ ] false-alarm batch کنترل‌شده (≈۵٪) در هر سه خانواده
- [ ] ریپوی ماژولار + config + `--resume` + README + تست
- [ ] هیچ مسیر سخت‌کد (`D:\...`) و اجرای یک‌دستوری
- [ ] همه‌ی ادعاها با CI/FDR و نتیجه‌ی تولیدشده‌ی خودکار
- [ ] نوت‌بوک نهایی تمیز و کوتاه (نمایش نتایج)
- [ ] گزارش صادقانه‌ی محدودیت‌ها

---

## ۷. فایل `CLAUDE.md` پیشنهادی (در ریشه‌ی ریپو بگذار)

```markdown
# Project: Silent failure detection of medical image classifiers under shift
Datasets (agreed with group): PathMNIST 128x128 (MedMNIST v2), MedMNIST-C v0.1.0
(corruptions applied at 224 then resized to 128), Camelyon17-WILDS v1.0 at 96x96.
Do NOT change dataset versions or resolutions without asking.

Rules:
- All hyperparameters live in configs/*.yaml. No hard-coded paths.
- Every stage is a script in scripts/ and resumable (--resume), logs to outputs/.
- Never fit references, thresholds, or fusion weights on any shifted evaluation data.
  Add assertions on uid overlap.
- Report every effect with a cluster-bootstrap 95% CI (>=2000 resamples); secondary
  families use BH-FDR. Verdict text must be generated from numbers.
- Seeds: 5 for core experiments. Set all RNG seeds.
- Run `python scripts/run_all.py --config configs/smoke.yaml` after any change.
- Prefer small, tested functions in sfd/. Notebook only displays results.
```

---

## ۸. پرامپت‌های آماده برای Claude Code (به ترتیب)

1. «`CLAUDE.md` و `ROADMAP.md` را بخوان. ساختار ریپو را مطابق بخش ۲ بساز، `configs/smoke.yaml` و `full.yaml` را بنویس، و `scripts/run_all.py` را با `--resume` بساز. هنوز مدل آموزش نده.»
2. «Stage 1 را پیاده کن: دانلود، split ثابت، leakage audit، Table 1 و assertهای نشت. با smoke تست کن.»
3. «Stage 2: ResNet-18، ۵ seed، temperature scaling، چک‌پوینت و resume. اول با smoke، بعد زمان اجرای full را تخمین بزن و به من بگو.»
4. «Stage 3: شیفت‌های A و B و C و X، مدل shortcut، جدول tidy پیش‌بینی.»
5. «Stage 4: سیگنال‌های uncertainty (MSP, entropy, energy)، representation (Mahalanobis class-conditional, kNN)، explanation (Grad-CAM + Grad-CAM++)، و validity check روی family C. جدول `monitoring_table.parquet`.»
6. «Stage 5: M0/M1/M2 با fusion monotone و held-out-family، bootstrap دو‌سطحی، jackknife، جدول contrastها و verdict خودکار.»
7. «Stage 6 و 7: batch monitoring و referral policy.»
8. «Stage 8 و 9: ablationها، FDR، Figure 1، نوت‌بوک نهایی، README. سپس کل پایپلاین را از صفر با smoke اجرا کن.»

بعد از هر مرحله از Claude Code بخواه: «تست‌ها را اجرا کن، خروجی‌ها را با exit criterion همان مرحله مقایسه کن و گزارش بده.»

---

## ۹. سؤال‌های باز برای گروه (کوتاه)
1. sub-dataset دقیق چیست؟
2. «۲ نتیجه‌ی مشخص» یعنی چه؟
3. فرمت تحویل/ارائه؟
4. آیا اجازه‌ی استفاده از معماری دوم و baselineهای اضافه به‌عنوان supplement داریم؟

---
*یادداشت صداقت علمی:* تخمین زمان آموزش و دانلود در این سند حدسی است و باید روز ۱ روی سخت‌افزار واقعی اندازه‌گیری شود. نتایج نسخه‌ی آیدا از خروجی ذخیره‌شده‌ی نوت‌بوکش خوانده شده و دوباره اجرا نشده است.

---

## ۱۰. اصلاحیه‌ی بعد از ویس‌های ابولفضل (جایگزین موارد متناقض بالا)

### تحویل نهایی = «۱ نوت‌بوک» که هم کد دارد و هم نتایج
- ابولفضل صریحاً گفته: یک فایل نوت‌بوک، کد و ریزالت‌ها داخلش، مرتب برای فرد سوم.
- پس **پوشه‌ی `sfd/` فقط محیط توسعه است**. خروجی نهایی `SilentFailure_Shayan.ipynb` است که:
  - هر Stage یک بخش با عنوان، یک پاراگراف توضیح، سلول کد، و خروجی (جدول/شکل) دارد؛
  - بالای نوت‌بوک یک بخش «چطور اجرا کنم + چه خروجی‌ای انتظار دارم + چقدر طول می‌کشد» دارد؛
  - کد تکراری و سلول‌های آزمایشی ندارد، و پایین آن یک جمع‌بندی نتایج با اعداد دارد؛
  - فقط یک فایل کمکی کنارش (مثلاً `requirements.txt`) و یک پوشه‌ی `outputs/` مرتب.
- روش پیشنهادی: کد را ماژولار بنویس و تست کن، سپس در Stage 9 نوت‌بوک نهایی را **از روی ماژول‌ها بساز** (کد لازم داخل سلول‌ها بیاید تا نوت‌بوک مستقل و خودکفا باشد).
- اولویت امتیاز: «تمیز و قابل‌فهم برای شخص سوم» دقیقاً همان نقطه‌ای است که ابولفضل به آیدا گفت ضعیف است.

### Sub-dataset (بدون مشخص‌شدن دقیق، باید با گروه نهایی شود)
ابولفضل فقط گفته «لازم نیست کل دیتاست باشد». پیشنهاد من، که باید با او و آیدا تأیید شود:
- **PathMNIST ۱۲۸px:** احتمالاً کامل هم روی لپ‌تاپ جا می‌شود (در RAM قابل نگهداری)؛ اگر لازم شد train را stratified به ~۵۰٪ کم کن، ولی val و test رسمی دست‌نخورده بمانند.
- **Camelyon17-WILDS:** نمونه‌برداری stratified و **گروه‌آگاه** (بر اساس بیمارستان/slide)، با seed ثابت، و ذخیره‌ی manifest (`subset_manifest.parquet`) تا هر دو نفر دقیقاً یک زیرمجموعه داشته باشید. هدف مثلاً چند ده هزار patch برای train و چند هزار برای هر بیمارستان OOD.
- MedMNIST-C: فقط روی تصاویر test/val زیرمجموعه اعمال شود.
- گزارش: جدول «اندازه‌ی کامل در برابر اندازه‌ی استفاده‌شده» + اینکه نمونه‌گیری چگونه بوده.

### سخت‌افزار: لپ‌تاپ ASUS TUF با RTX 3060
- لپ‌تاپ‌های TUF Gaming با 3060 معمولاً **RTX 3060 Laptop با ۶GB VRAM** دارند. با `nvidia-smi` چک کن. ۶GB برای این پروژه **کافی است** (ResNet-18 روی ۱۲۸px با batch ۱۲۸ و mixed precision حدود ۲–۳GB). سرعتش هم از T4 کولب کمتر نیست.
- گلوگاه واقعی معمولاً **دیسک، RAM و ویندوز** است نه GPU:
  - فضای خالی کافی برای Camelyon (ده‌ها GB پس از استخراج)؛
  - روی ویندوز `num_workers>0` در PyTorch مشکل‌ساز است (نوت‌بوک آیدا هم آن را ۰ گذاشته بود و کند می‌شود). راه‌حل: داده‌ی ۱۲۸px را در RAM/روی GPU نگه دار و augmentation را روی GPU انجام بده، یا از **WSL2 (Ubuntu)** استفاده کن؛
  - استفاده از `torch.autocast` (AMP) و `channels_last`؛
  - لپ‌تاپ به برق وصل و حالت Turbo/Performance فعال، خنک‌کاری مناسب برای اجرای چندساعته؛
  - همه‌چیز resumable باشد تا قطع شدن برق/خواب لپ‌تاپ کار را خراب نکند.
- اگر ۵ seed کامل برای همه‌ی مدل‌ها در زمان نگنجید: ۵ seed برای ResNet-18 اصلی، و معماری دوم/ensemble فقط ۱–۲ seed.

### Stage 0 اضافه
- `nvidia-smi` و `torch.cuda.get_device_properties(0)` را در `env_record.json` ثبت کن.
- یک benchmark کوتاه: زمان یک epoch ResNet-18 روی PathMNIST ۱۲۸px — با آن برنامه‌ی زمانی را دقیق کن.
