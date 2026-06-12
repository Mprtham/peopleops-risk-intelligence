# PeopleOps Analytics Platform

> End-to-end HR analytics — attrition prediction, pay equity analysis, and automated retention actions — built on BigQuery, dbt, and Power BI.

<!-- CI badges — fill in your repo path -->
![dbt CI](https://github.com/YOUR_ORG/peopleops-analytics/actions/workflows/dbt_ci.yml/badge.svg)
![Power BI BPA](https://github.com/YOUR_ORG/peopleops-analytics/actions/workflows/pbi_bpa.yml/badge.svg)
![Gated Refresh](https://github.com/YOUR_ORG/peopleops-analytics/actions/workflows/refresh.yml/badge.svg)

---

## The problem

Replacing an employee costs roughly one times their annual salary — recruiting fees, lost productivity, onboarding, and time to full effectiveness. For a 4,000-person organisation paying a median £45,000, a 15% annual attrition rate represents roughly **£27 million in replacement cost per year**. Most of that is preventable: people leave for identifiable, addressable reasons — salary stagnation, no promotion, workload, low engagement.

The typical response is a quarterly exit survey. By then it is too late.

---

## What I built

A production-grade analytics system that identifies employees at risk of leaving **before** they decide to go, and prescribes the right intervention for each driver.

```
generate_synthetic_data.py  (seeded, reproducible — 4,000 employees, 3 years)
        │
        ▼
BigQuery  peopleops_raw  (5 tables: employees, role_history, reviews, absences, exits)
        │
        ▼
dbt  staging → marts
        ├── dim_employee          (SCD Type 2 — real change history)
        ├── fct_attrition_snapshots  (point-in-time features, leakage-proof)
        ├── fct_absence_monthly
        └── rpt_pay_equity        (gender × dept × role band, gap %)
        │
        ├──▶ Power BI (PBIP in Git, BPA in CI, gated REST API refresh)
        ├──▶ ml/train_model.py    (logistic regression + SHAP, time-split)
        │         └──▶ ml/risk_api.py  (FastAPI  POST /predict)
        └──▶ ml/retention_playbook.md
```

---

## Why this is different from a typical HR project

| Dimension | Typical approach | This project |
|---|---|---|
| Data | IBM Kaggle CSV, static | Custom synthetic event data with change history |
| Pipeline | pandas notebook | BigQuery + dbt, tested and documented |
| Model | Black-box XGBoost, leaky features | Point-in-time logistic regression + SHAP |
| Dashboard | Static screenshots | Power BI PBIP in Git, CI-validated, API-refreshed |
| Output | Confusion matrix | Risk API + tiered retention playbook + pay-equity report |

---

## Results

| Metric | Value |
|---|---|
| Annualised attrition (synthetic cohort) | 15.3% |
| dbt tests passing | 75/75 |
| Model ROC-AUC — time-split test (2025 snapshots) | **0.665** |
| Model ROC-AUC — same-distribution diagnostic | 0.681 |
| Model PR-AUC (test) | **0.462** (vs 0.156 random baseline — 3× lift) |
| Precision@top-10% | **55.1%** of flagged employees actually leave |
| API response time (cached BQ client) | < 300 ms |

The dominant SHAP drivers are `avg_rating_all` (OR=0.65: lower rating = higher risk) and `absence_events_per_month` (OR=1.18: higher absence rate = higher risk) — exactly the causal factors encoded in the synthetic data, which confirms the model is finding real signal.

**On AUC**: the time-split gives 0.665 vs 0.681 on a same-distribution diagnostic split — a gap of only 0.016, meaning distributional shift between train (2024 and earlier) and test (2025 snapshots) is small. The ceiling is set by label noise: even the highest-risk employees have only a 20-30% chance of exiting in any given 6-month window, so the binary label is inherently noisy. Industry benchmarks for production attrition models are 0.65–0.80; this project sits at the bottom of that range with a logistic regression and observable HR features, which is honest and appropriate.

---

## How the automation works

### Gated Power BI refresh

The Power BI dashboard **cannot refresh on bad data**. The Monday morning workflow is:

```
dbt build (run + test)
    │
    ├─ any test fails → workflow exits, no refresh triggered
    │
    └─ all tests pass → Power BI REST API refresh called
```

This means a broken dbt model or a data integrity failure (e.g. absence events post-dating an exit) blocks the dashboard from updating — not silently wrong data that a stakeholder might act on.

### BPA in CI

Every pull request touching `powerbi/` runs Tabular Editor 2's Best Practice Analyzer against the semantic model. The 12 rules enforce: all measures documented, no implicit measures on fact tables, percentage measures formatted as percentages, no `IFERROR` wrappers masking errors, no bi-directional relationships without justification.

A PR with an undocumented measure or a badly formatted KPI **cannot merge**.

### Risk API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"employee_id": 42}'
```

```json
{
  "employee_id": 42,
  "snapshot_date": "2025-12-31",
  "risk_score": 0.7341,
  "risk_tier": "high",
  "top_factors": [
    {"feature": "avg_rating_all",            "shap_effect": 0.334, "raw_value": 2.1,  "direction": "decreases_risk"},
    {"feature": "absence_events_per_month",  "shap_effect": 0.117, "raw_value": 1.4,  "direction": "increases_risk"},
    {"feature": "absence_events_last90",     "shap_effect": 0.041, "raw_value": 4.0,  "direction": "increases_risk"}
  ],
  "playbook_action": "Stay interview within 5 working days (HRBP owner)"
}
```

The API returns not just a score but the specific playbook action that applies, derived from the same thresholds in `ml/retention_playbook.md`. The model output is an input to a human conversation — not an automated action.

---

## Retention playbook (excerpt)

| Risk score | Driver | Owner | Action | SLA |
|---|---|---|---|---|
| > 0.70 | Any | HRBP | Stay interview | 5 working days |
| 0.50–0.70 | No promotion 2+ years | HRBP + Line manager | Promotion committee flag | 10 working days |
| 0.50–0.70 | Absence days > 10 (90d) | Line manager | Workload review | 5 working days |
| 0.40–0.70 | Salary < −10% vs median | Compensation | Off-cycle salary review | Next comp cycle |

Full playbook including model governance, fairness checks, and retrain cadence: [`ml/retention_playbook.md`](ml/retention_playbook.md)

---

## Pay equity analysis

`rpt_pay_equity` computes median salary by gender × department × role level, with the gap relative to the all-gender median for that band.

| Department | Role level | Gender | Median salary | Gap vs band median |
|---|---|---|---|---|
| Engineering | 2 — Senior | Female | £61,200 | −4.2% |
| Engineering | 2 — Senior | Male | £64,800 | +1.4% |
| Engineering | 2 — Senior | Non-binary | £60,500 | −5.3% |

Any band with `pay_gap_pct` outside ±5% triggers a mandatory compensation review before the next salary cycle closes — enforced by the playbook, surfaced in the dashboard.

---

## Running it yourself

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Google Cloud SDK
gcloud auth application-default login
export GCP_PROJECT=your-project-id

# dbt profile (copy and fill in project ID)
cp dbt/profiles.yml.example dbt/profiles.yml
```

### End-to-end

```bash
make data       # generate 4,000 employees × 3 years of event data
make load       # load CSVs into BigQuery (peopleops_raw)
make transform  # dbt build — staging + marts, 15+ tests
make train      # logistic regression + SHAP, saves artifacts to ml/artifacts/
make api        # FastAPI on localhost:8000
make docs       # dbt docs site on localhost:8080
```

### dbt only

```bash
make check      # dbt test (without re-running models)
```

---

## Architecture decisions

**Why logistic regression, not XGBoost?**
Interpretability. An HRBP presenting a "stay interview" to an employee needs to explain why. SHAP on a logistic regression produces odds ratios that a non-technical stakeholder can understand and challenge. A gradient-boosted model with 0.02 better AUC is not worth the trust cost.

**Why point-in-time snapshots?**
The classic HR model leakage mistake: using features computed at exit time (e.g. "performance rating" measured after the employee handed in notice). `fct_attrition_snapshots` joins features strictly as of each quarter-end snapshot date. The label (`exited_within_6m`) looks forward from that date — that is the target, not leakage.

**Why Power BI PBIP in Git?**
A dashboard that lives only in a workspace cannot be code-reviewed, rolled back, or CI-validated. PBIP format makes the semantic model (`model.bim`) and report definition diff-able. Combined with BPA on PRs, it brings the same engineering discipline to the BI layer that dbt brings to the SQL layer.

**Why a gated refresh?**
A stale dashboard is better than a wrong one. If `fct_attrition_snapshots` has bad data — a broken join, a missing snapshot quarter — the dashboard should not silently serve incorrect attrition rates to leadership. The `dbt build` gate ensures that only clean, tested data reaches Power BI.

---

## Repository structure

```
peopleops-analytics/
├── data/generate_synthetic_data.py     # seeded event data, 4,000 employees
├── ingestion/load_to_bigquery.py       # explicit schemas, WRITE_TRUNCATE
├── dbt/
│   ├── models/staging/                 # 5 staging views
│   ├── models/marts/                   # dim, 2 facts, 1 report
│   └── tests/assert_no_events_after_exit.sql
├── ml/
│   ├── train_model.py                  # LR + SHAP, time-split, PR-AUC
│   ├── risk_api.py                     # FastAPI /predict + /health
│   └── retention_playbook.md           # tiers, actions, governance
├── powerbi/
│   ├── PeopleOps.pbip                  # PBIP project file
│   ├── PeopleOps.SemanticModel/        # model.bim — 10 DAX measures
│   └── bpa_rules.json                  # 12 BPA rules for CI
├── .github/workflows/
│   ├── dbt_ci.yml                      # dbt build + test on PR
│   ├── pbi_bpa.yml                     # Tabular Editor BPA on PR
│   └── refresh.yml                     # gated Monday refresh
├── Makefile
└── requirements.txt                    # pinned versions
```

---

## Demo

*[2-minute walkthrough video — link to be added after recording]*

The demo covers: running `make all`, the dbt test output, a live API call, the Power BI dashboard with department drill-down and pay equity page, and a simulated test failure blocking the refresh.
