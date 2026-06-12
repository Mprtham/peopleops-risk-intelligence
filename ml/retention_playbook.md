# Retention Playbook

**Version:** 1.0  
**Owner:** People Operations  
**Model:** Logistic regression on `fct_attrition_snapshots` (retrained quarterly)

---

## Intervention tiers

| Risk score | Owner | Action | SLA |
|---|---|---|---|
| > 0.70 | HRBP | Stay interview scheduled | 5 working days |
| 0.50–0.70 + no promotion in 2+ years | HRBP + Line manager | Flag for promotion committee at next cycle | 10 working days |
| 0.50–0.70 + absence_days_last90 > 10 | Line manager | Workload review + wellbeing check-in | 5 working days |
| 0.40–0.70 + salary_vs_median_pct < −0.10 | Compensation team | Compensation review against band | Next comp cycle |
| 0.40–0.70 + avg_rating_last2 < 2.5 | Line manager | Performance support plan | 10 working days |

---

## Stay interview guide (risk > 0.70)

Ask in this order — do not open with the salary question:

1. "What's been the most energising part of your work recently?"
2. "Is there anything making your day-to-day harder than it needs to be?"
3. "Where would you like to be in 18 months — what would make that possible here?"
4. "Is there anything we could change that would make a meaningful difference for you?"

Document responses in the HRIS within 48 hours. Escalate to VP People if no feasible retention action exists.

---

## Driver-to-action map

| Top SHAP driver | Root cause | Action |
|---|---|---|
| `months_in_current_role` high | Stagnation — no growth signal | Accelerated development plan; stretch project |
| `salary_vs_median_pct` negative | Underpaid vs peers | Off-cycle salary review; equity adjustment |
| `avg_rating_last2` low | Disengagement or performance gap | Clarify expectations; coaching or PIP |
| `absence_days_last90` high | Burnout or health issue | Workload audit; EAP referral |
| `tenure_months` low (< 12) | New-joiner experience | 90-day check-in; buddy programme review |

---

## Pay equity escalation

`rpt_pay_equity` is reviewed quarterly. Any `pay_gap_pct` outside ±5% for the same department × role level triggers a mandatory compensation review before the next salary cycle closes.

Protected characteristics covered: gender. Future cycles: ethnicity, disability status (pending data collection consent process).

---

## Model governance

### Retrain cadence
- Full retrain: quarterly (after each quarter-end snapshot is added to `fct_attrition_snapshots`)
- Trigger: `make train` in CI after `dbt build` passes
- Retrain is blocked if `roc_auc_test` drops below 0.70 vs the previous run — escalate to data team

### Fairness check (every retrain)
Run the fairness audit section of `train_model.py`. Flag if mean risk score differs by more than 0.05 across gender groups after controlling for role level and department. Disproportionate scores without a business-justified driver must be investigated before the model is promoted.

### Leakage guard
AUC > 0.95 on the test set is treated as a leakage signal, not a success. `train_model.py` prints a warning and the release is blocked.

### Features excluded by policy
- `gender` — present in data, excluded from model inputs to avoid disparate impact. Used only in fairness audit.
- `date_of_birth` / `age_at_snapshot` — included (age is a legitimate tenure-correlated predictor) but monitored for age discrimination in action rates.

### Human-in-the-loop
Model output is an input to a conversation — not an automated action. No employee is contacted, performance-managed, or compensated based on a score alone. All interventions require line manager or HRBP sign-off.
