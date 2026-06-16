# PeopleOps — HR Attrition Intelligence

> End-to-end ML pipeline for predicting employee attrition, with SHAP explainability and Power BI embedded reporting.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![BigQuery](https://img.shields.io/badge/BigQuery-Data_Warehouse-4285F4?style=flat-square&logo=googlebigquery&logoColor=white)](https://cloud.google.com/bigquery)
[![dbt](https://img.shields.io/badge/dbt-Transforms-FF694B?style=flat-square&logo=dbt&logoColor=white)](https://getdbt.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-7F52FF?style=flat-square)](https://shap.readthedocs.io)

[Case study →](https://portfolio-iota-taupe-34.vercel.app/work/peopleops)

---

## What it does

PeopleOps transforms raw HRIS data into an attrition prediction system that HR teams can actually act on. It tells you _who_ is likely to leave, _why_ (via SHAP), and _how much_ each departure costs — surfaced in a Power BI dashboard with drill-through to individual explanations.

**Key outcomes**
- 87% recall on high-risk employees (XGBoost + feature engineering)
- SHAP waterfall charts embedded per employee in Power BI PBIP format
- Retention intervention cost modelled at £18K avg saving per prevented departure
- dbt transforms from raw HRIS → feature store → model input: fully tested + documented

---

## Architecture

```
HRIS / ATS
     ↓
[BigQuery raw layer]  →  dbt staging models
     ↓
[dbt feature store]  →  tenure, pay-band delta, manager change rate, …
     ↓
[XGBoost model]  →  attrition probability + SHAP values
     ↓
[Power BI PBIP]  →  risk dashboard + individual SHAP waterfall
```

---

## Feature engineering highlights

| Feature | Signal |
|---|---|
| `pay_band_delta_12m` | Pay change relative to peer band over 12 months |
| `manager_change_count` | Number of reporting-line changes (high = instability signal) |
| `skip_level_ratio` | Skip-level meetings attended vs peers (engagement proxy) |
| `promotion_gap_months` | Months since last promotion vs role-level median |
| `tenure_band` | Bucketed: 0-6m, 6-18m, 18-36m, 36m+ (J-curve attrition pattern) |
| `team_attrition_rate_90d` | Rolling 90-day attrition in immediate team |

---

## Stack

- **Data warehouse**: Google BigQuery
- **Transforms**: dbt Core (staging → intermediate → feature store → model input)
- **Model**: XGBoost with Optuna hyperparameter tuning, 5-fold CV
- **Explainability**: SHAP (TreeExplainer) — waterfall, beeswarm, force plots
- **BI**: Power BI PBIP (semantic model + report separated), embedded SHAP PNGs
- **Orchestration**: Cloud Composer (Airflow) DAG, weekly refit

---

## Quick start

```bash
git clone https://github.com/Mprtham/peopleops-attrition
cd peopleops-attrition

# dbt setup
pip install dbt-bigquery
cp profiles.example.yml ~/.dbt/profiles.yml   # add GCP credentials
dbt deps && dbt build

# Train model
python ml/train.py --target attrition_90d --experiment baseline_v1

# Generate SHAP report
python ml/explain.py --model runs/baseline_v1/model.pkl --output reports/shap/
```

---

## Project structure

```
peopleops-attrition/
├── dbt/
│   ├── models/
│   │   ├── staging/        # Raw HRIS source models
│   │   ├── intermediate/   # Joins + business logic
│   │   └── feature_store/  # ML-ready feature tables
│   └── tests/              # dbt schema + custom tests
├── ml/
│   ├── train.py            # XGBoost training + MLflow logging
│   ├── explain.py          # SHAP value computation + plot export
│   └── evaluate.py         # Calibration, recall@k, business metrics
├── powerbi/                # PBIP semantic model + report files
└── reports/                # Generated SHAP outputs
```
