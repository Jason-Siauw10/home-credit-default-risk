# Home Credit Default Risk Prediction
**Virtual Internship Experience — Rakamin Academy × Home Credit**

---

## Problem Statement

Home Credit serves customers who are underserved by traditional banking — people with little or no formal credit history who are typically rejected by conventional lenders. The challenge is finding the right balance:

- **Too strict** → creditworthy customers rejected → lost revenue + financial exclusion
- **Too loose** → high-risk customers approved → loan defaults → financial loss

**Goal:** Build a model that predicts the probability of payment difficulty (`TARGET = 1`) so that loans can be offered with terms tailored to each applicant's actual repayment capacity — rather than being rejected outright.

---

## Repository Structure

```
home-credit-default-risk/
├── home_credit_default_risk.ipynb   ← Main notebook (full end-to-end pipeline)
├── submission.csv                   ← Final test set predictions
├── README.md                        ← This file
└── data/                            ← Not tracked (download from Kaggle)
    ├── application_train.csv
    ├── application_test.csv
    ├── bureau.csv
    ├── bureau_balance.csv
    ├── previous_application.csv
    ├── POS_CASH_balance.csv
    ├── credit_card_balance.csv
    ├── installments_payments.csv
    └── HomeCredit_columns_description.csv
```

---

## Dataset

Source: [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

| Table | Rows (approx) | Description |
|---|---|---|
| `application_train/test.csv` | 307,511 / 48,744 | Main table — applicant demographics, loan amounts, EXT_SOURCE scores |
| `bureau.csv` | ~1.7M | Credits at other institutions reported to Credit Bureau |
| `bureau_balance.csv` | ~27M | Monthly DPD status for each bureau credit |
| `previous_application.csv` | ~1.7M | Past loan applications at Home Credit |
| `POS_CASH_balance.csv` | ~10M | Monthly balances of previous POS/cash loans |
| `credit_card_balance.csv` | ~3.8M | Monthly credit card behaviour at Home Credit |
| `installments_payments.csv` | ~13.6M | Actual repayment history — paid vs due amount |

All supplementary tables join to the main table via `SK_ID_CURR`.

---

## Notebook Pipeline

### Step 1 — Load Data & Column Descriptions
- All 8 tables loaded
- `describe_col()` helper for looking up any column from `HomeCredit_columns_description.csv`

### Step 2 — Exploratory Data Analysis
- Missing value analysis (67 columns with missing, 17 dropped at >60%)
- Target distribution: **91.93% no difficulty / 8.07% payment difficulty**
- Numerical distributions, bivariate analysis, categorical default rates
- Correlation analysis — `EXT_SOURCE` scores are the strongest predictors

**Business Insight 1 — Employment Duration:**
Applicants with <1 yr employment default at **10.97%** — nearly 3× the rate of those employed >20 years (4.19%).
→ *Action: Require >1yr employment or co-signer; offer pre-approval financial literacy program*

**Business Insight 2 — Debt-Burden Ratio:**
Default rate rises from 6.47% (annuity <5% of income) to 8.76% (annuity 20–30% of income).
→ *Action: Cap monthly annuity at ≤20% of verified income; extend tenure for borderline cases*

### Step 3 — Data Preprocessing & Feature Engineering
- **Domain-driven aggregation** of 5 supplementary tables (not generic mean/sum/max/min of everything)
- **15+ engineered features**: credit ratios, time-based stability indicators, EXT score combinations (`EXT_MEAN`, `EXT_PROD`), interaction terms (`EXT2_X_AGE`, `EXT2_X_EMPLOYED`)
- `pd.factorize` encoding on 16 categorical columns (train + test combined to prevent unseen category errors)
- 25 columns dropped for >60% missing; median imputation using training set only
- **Feature selection**: Domain-protect list (20 delinquency/context signals) → variance filter → correlation filter → signal filter → tree importance → **80 final features**
- Samples-to-features ratio: **3,075:1**

### Step 4 — Model Training

| Model | Key Parameters | Purpose |
|---|---|---|
| Logistic Regression | `C=0.1`, `solver=saga`, `class_weight=balanced` | Compliance / scorecard baseline |
| Random Forest | `n_estimators=200`, `max_depth=12`, `class_weight=balanced_subsample` | Ensemble baseline |
| HistGradientBoosting | `max_iter=500`, `lr=0.05`, `max_leaf_nodes=63`, `class_weight=balanced` | Best performance model |

All models use `class_weight='balanced'` to handle the 8.07% class imbalance.

### Step 5 — Model Evaluation

| Model | Val ROC-AUC | Recall (Bad) | Precision (Bad) | Train-Val Gap |
|---|---|---|---|---|
| Logistic Regression | 0.7587 | 0.68 | 0.17 | -0.001 ✓ |
| Random Forest | 0.7613 | 0.59 | 0.20 | 0.086 ⚠ |
| **HistGradientBoosting** | **0.7788** | **0.67** | **0.19** | 0.064 |

**Primary metric:** ROC-AUC (ranking-based, robust to class imbalance)

**Top features by permutation importance:**
1. `EXT_MEAN` (0.067) — external credit scores are the dominant signal
2. `CREDIT_ANNUITY_RATIO` (0.011) — debt burden directly confirms Insight 2
3. `CREDIT_GOODS_GAP`, `POS_MONTHS_COUNT`, `CODE_GENDER` — domain-protected features validate the protection strategy

### Step 6 — Business Impact & Recommendations

At decision threshold = 0.5, per 100,000 annual applications:

| Model | Losses Prevented | Revenue Lost | Net Benefit |
|---|---|---|---|
| **HistGradientBoosting ★** | IDR 1,453M | IDR 700M | **IDR 753M** |
| Random Forest | IDR 1,274M | IDR 582M | IDR 692M |
| Logistic Regression | IDR 1,481M | IDR 825M | IDR 656M |

*Assumptions: avg loan IDR 599,026 · loss given default 45% · profit margin 5%*

---

## How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/home-credit-default-risk.git
   cd home-credit-default-risk
   ```

2. **Download the data** from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data) and place all CSV files in a `data/` folder (or the same directory as the notebook).

3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   # Optional for faster gradient boosting:
   pip install lightgbm
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook home_credit_default_risk.ipynb
   ```
   > Update `DATA_DIR` at the top of the notebook if your CSVs are in a subfolder.

---

## Key Findings

1. **External credit scores dominate** — `EXT_MEAN` alone causes a 0.067 AUC drop when removed. Third-party credit history is the most informative signal available and cannot be replaced by internal data alone.

2. **Debt burden is actionable** — `CREDIT_ANNUITY_RATIO` ranks 2nd in permutation importance, directly confirming Insight 2. Home Credit controls this ratio at loan origination.

3. **HistGradientBoosting is the best model** — AUC 0.7788, ~25× faster than sklearn GradientBoosting, manageable overfitting gap (0.064).

4. **Random Forest overfits moderately** (gap 0.086) — production deployment would benefit from `max_depth=8`, `min_samples_leaf=60`.

5. **Logistic Regression retained for compliance** — its coefficients can be mapped directly to a credit scorecard for regulatory reporting.

---

## Business Recommendations

| # | Recommendation | Basis |
|---|---|---|
| 1 | Risk-tiered loan terms (score < 0.2 / 0.2–0.4 / > 0.4) | Model probability scores |
| 2 | Require >1yr employment or co-signer for new entrants | Insight 1 |
| 3 | Cap monthly annuity at ≤20% of verified income | Insight 2 |
| 4 | Invest in EXT_SOURCE bureau partnerships | Feature importance |
| 5 | Keep Logistic Regression for regulatory scorecard | Interpretability |
| 6 | Deploy HistGB as REST API; re-train quarterly | Operational |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn >= 1.0
lightgbm (optional — falls back to HistGradientBoostingClassifier)
```

---

*Rakamin Academy Virtual Internship Experience — Home Credit Indonesia*
