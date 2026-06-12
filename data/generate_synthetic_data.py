"""
Synthetic HR data generator for PeopleOps Analytics Platform.
All randomness seeded for reproducibility. Outputs five CSVs to data/raw/.

Causal structure: each employee has a persistent `individual_risk` score
drawn from Beta(2, 6) (~mean 0.25, right-skewed). This latent variable
simultaneously drives lower ratings, more absences, slower promotions, and
higher exit probability — creating consistent feature-outcome signal across
all 12 snapshot dates so the ML model can achieve AUC 0.75-0.90.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)
N_EMPLOYEES = 4000
OUTPUT_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance", "Operations"]

ROLE_LADDER = {
    "Sales":       ["Sales Associate", "Sales Executive", "Senior Sales Executive", "Sales Manager", "Sales Director"],
    "Engineering": ["Junior Engineer", "Engineer", "Senior Engineer", "Engineering Manager", "Engineering Director"],
    "HR":          ["HR Coordinator", "HR Specialist", "Senior HR Specialist", "HR Manager", "HR Director"],
    "Marketing":   ["Marketing Associate", "Marketing Specialist", "Senior Marketer", "Marketing Manager", "Marketing Director"],
    "Finance":     ["Finance Analyst", "Senior Analyst", "Finance Manager", "Senior Finance Manager", "Finance Director"],
    "Operations":  ["Operations Associate", "Operations Specialist", "Senior Operations Specialist", "Operations Manager", "Operations Director"],
}

# Base salary ranges by role level (0=entry, 4=director)
SALARY_BANDS = [
    (28_000, 38_000),
    (38_000, 55_000),
    (55_000, 75_000),
    (75_000, 105_000),
    (105_000, 160_000),
]

CONTRACT_TYPES = ["Permanent", "Permanent", "Permanent", "Fixed-term", "Part-time"]
GENDERS = ["Male", "Female", "Non-binary"]
GENDER_WEIGHTS = [0.47, 0.47, 0.06]

ABSENCE_TYPES = ["Sick", "Personal", "Family", "Medical", "Other"]
EXIT_REASONS = ["Resignation", "Resignation", "Resignation", "Redundancy", "Performance", "Retirement"]


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def salary_for_level(level: int, dept: str) -> int:
    lo, hi = SALARY_BANDS[level]
    if dept in ("Engineering", "Finance"):
        lo = int(lo * 1.1)
        hi = int(hi * 1.1)
    return random.randint(lo, hi)


def next_anniversary(d: date, months: int) -> date:
    """Add N months to d, clamping end-of-month dates (e.g. Feb 29 -> Feb 28)."""
    year  = d.year  + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    try:
        return date(year, month, d.day)
    except ValueError:
        # Day out of range for month (Feb 28/29)
        import calendar
        return date(year, month, calendar.monthrange(year, month)[1])


# ---------------------------------------------------------------------------
# 1. Employees — includes individual_risk latent variable
# ---------------------------------------------------------------------------

def build_employees() -> pd.DataFrame:
    """
    individual_risk ~ Beta(2, 6): mean=0.25, sd=0.146.
    Wider spread than previous Beta(1.5, 5) → more extreme employees.
    """
    # Beta(1,3): mean=0.25, sd=0.194 — wider spread than Beta(2,6), giving
    # clearer separation between low-risk and high-risk employees
    rng = np.random.default_rng(SEED)
    individual_risks = rng.beta(1, 3, size=N_EMPLOYEES)

    rows = []
    for i, eid in enumerate(range(1, N_EMPLOYEES + 1)):
        gender = random.choices(GENDERS, weights=GENDER_WEIGHTS)[0]
        if gender == "Male":
            name = fake.name_male()
        elif gender == "Female":
            name = fake.name_female()
        else:
            name = fake.name()

        dept = random.choice(DEPARTMENTS)
        hire_date = random_date(START_DATE - timedelta(days=3 * 365), END_DATE - timedelta(days=60))
        dob = hire_date - timedelta(days=random.randint(22 * 365, 55 * 365))

        rows.append({
            "employee_id": eid,
            "full_name": name,
            "gender": gender,
            "date_of_birth": dob,
            "department": dept,
            "hire_date": hire_date,
            "contract_type": random.choices(CONTRACT_TYPES)[0],
            "individual_risk": float(individual_risks[i]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Role history — high-risk employees promoted much less frequently
# ---------------------------------------------------------------------------

def build_role_history(employees: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, emp in employees.iterrows():
        eid = emp["employee_id"]
        dept = emp["department"]
        hire_date = emp["hire_date"]
        risk = emp["individual_risk"]
        ladder = ROLE_LADDER[dept]

        current_level = 0
        current_salary = salary_for_level(0, dept)
        effective_from = hire_date

        check_date = hire_date
        segments = []
        while check_date <= END_DATE:
            tenure_years = (check_date - hire_date).days / 365.25
            if current_level < 4:
                # High-risk (0.8) gets ~20% of base promotion probability
                # Low-risk  (0.1) gets ~92% of base promotion probability
                promo_multiplier = 1.0 - risk * 0.85
                promo_prob = min(0.25 * tenure_years, 0.40) / 12 * promo_multiplier
                if random.random() < promo_prob:
                    segments.append((effective_from, check_date - timedelta(days=1), current_level, current_salary))
                    current_level = min(current_level + 1, 4)
                    current_salary = max(current_salary + random.randint(3_000, 12_000),
                                        salary_for_level(current_level, dept))
                    effective_from = check_date
            elif (check_date - effective_from).days >= 365 and random.random() < 0.7:
                segments.append((effective_from, check_date - timedelta(days=1), current_level, current_salary))
                current_salary += random.randint(500, 4_000)
                effective_from = check_date
            check_date += timedelta(days=30)

        segments.append((effective_from, None, current_level, current_salary))

        for ef, et, lvl, sal in segments:
            rows.append({
                "employee_id": eid,
                "job_role": ladder[lvl],
                "role_level": lvl,
                "salary": sal,
                "effective_from": ef,
                "effective_to": et,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Performance reviews — semi-annual, high-risk rated much lower
# ---------------------------------------------------------------------------

def build_performance_reviews(employees: pd.DataFrame) -> pd.DataFrame:
    """
    Reviews every 6 months starting at 6-month anniversary.
    High-risk employees get ratings centred around 2.0-2.5 (mean=4.5 - risk*3.0),
    low-risk employees get ratings around 4.0-4.5.
    Tight std=0.5 means the signal clearly dominates the noise.
    """
    rows = []
    review_id = 1

    for _, emp in employees.iterrows():
        eid = emp["employee_id"]
        hire_date = emp["hire_date"]
        risk = emp["individual_risk"]

        # risk=0.0->5.5(capped 5), risk=0.25->4.5, risk=0.5->3.5, risk=0.75->2.5, risk=1->1.5
        # std=0.3 so avg of 6 reviews has noise <0.13 — signal clearly dominates
        rating_mean = 5.5 - risk * 4.0

        review_date = next_anniversary(hire_date, 6)   # first review at 6 months
        offset = 6
        while review_date <= END_DATE:
            rating = int(np.clip(np.round(np.random.normal(rating_mean, 0.3)), 1, 5))
            rows.append({
                "review_id": review_id,
                "employee_id": eid,
                "review_date": review_date,
                "rating": rating,
            })
            review_id += 1
            offset += 6
            review_date = next_anniversary(hire_date, offset)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Absence events — Poisson process, high-risk employees far more absent
# ---------------------------------------------------------------------------

def build_absence_events(employees: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly absence rate: low-risk (0.1) ~0.25/month = 3/year,
                          high-risk (0.8) ~1.10/month = 13/year.
    Using a per-month Poisson draw so the signal is consistent in any 90-day window.
    """
    rng = np.random.default_rng(SEED + 2)
    rows = []
    absence_id = 1

    for _, emp in employees.iterrows():
        eid = emp["employee_id"]
        hire_date = emp["hire_date"]
        risk = emp["individual_risk"]

        # Monthly Poisson rate — low (0.1): 0.32/mo=3.8/yr, high (0.8): 1.62/mo=19.4/yr
        monthly_rate = 0.18 + risk * 1.80

        current_date = max(hire_date, START_DATE)
        while current_date <= END_DATE:
            n_events = int(rng.poisson(monthly_rate))
            for _ in range(n_events):
                offset_days = int(rng.integers(0, 30))
                absence_date = current_date + timedelta(days=offset_days)
                if absence_date > END_DATE:
                    break
                rows.append({
                    "absence_id": absence_id,
                    "employee_id": eid,
                    "absence_date": absence_date,
                    "days": int(rng.choice([1, 2, 3, 5], p=[0.5, 0.25, 0.15, 0.1])),
                    "absence_type": random.choice(ABSENCE_TYPES),
                })
                absence_id += 1
            current_date += timedelta(days=30)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Exits — driven directly by individual_risk
# ---------------------------------------------------------------------------

def build_exits(employees: pd.DataFrame) -> pd.DataFrame:
    """
    annual_p = clip(0.04 + individual_risk * 0.46, 0.03, 0.50)

    High-risk (0.8) -> 41% annual exit probability
    Low-risk  (0.1) -> 9%  annual exit probability
    Mean (0.25)     -> 16% -> ~15% realized attrition after window effects

    Since individual_risk also drives lower ratings, more absences, slower
    promotions, the features at EVERY snapshot date correlate with exit outcome.
    """
    obs_start = pd.Timestamp(START_DATE)
    obs_end = pd.Timestamp(END_DATE)
    hire_dates = pd.to_datetime(employees.set_index("employee_id")["hire_date"]).to_dict()
    risk_dict = employees.set_index("employee_id")["individual_risk"].to_dict()

    rng = np.random.default_rng(SEED + 1)

    rows = []
    exit_id = 1
    for eid in employees["employee_id"]:
        risk = risk_dict[eid]
        annual_p = float(np.clip(0.04 + risk * 0.44, 0.03, 0.40))
        monthly_p = 1.0 - (1.0 - annual_p) ** (1.0 / 12.0)

        hire_ts = pd.Timestamp(hire_dates[eid])
        window_start = max(hire_ts, obs_start)
        window_months = max(int((obs_end - window_start).days / 30.44), 1)

        months_to_exit = int(rng.geometric(p=monthly_p))
        if months_to_exit > window_months:
            continue  # survived the observation window

        exit_ts = window_start + pd.Timedelta(days=int(months_to_exit * 30.44) + int(rng.integers(0, 14)))
        if exit_ts > obs_end:
            continue

        rows.append({
            "exit_id": exit_id,
            "employee_id": int(eid),
            "exit_date": exit_ts.date(),
            "exit_reason": random.choices(
                EXIT_REASONS,
                weights=[0.55, 0.55, 0.55, 0.15, 0.10, 0.05],
            )[0],
        })
        exit_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating employees (individual_risk ~ Beta(1,3))...")
    employees = build_employees()

    print("Generating role history (80%% promotion penalty for high-risk)...")
    role_history = build_role_history(employees)

    print("Generating performance reviews (semi-annual, risk-stratified ratings)...")
    reviews = build_performance_reviews(employees)

    print("Generating absence events (Poisson process, 6.6x rate for high-risk)...")
    absences = build_absence_events(employees)

    print("Generating exits (annual_p = 0.04 + risk*0.44, max 40%)...")
    exits = build_exits(employees)

    # Enforce: no event after exit date
    exit_dates = exits.set_index("employee_id")["exit_date"].to_dict()

    def clip_to_exit(df, date_col):
        return df[df.apply(
            lambda r: exit_dates.get(r["employee_id"]) is None or r[date_col] <= exit_dates.get(r["employee_id"]),
            axis=1
        )]

    reviews = clip_to_exit(reviews, "review_date")
    absences = clip_to_exit(absences, "absence_date")

    # Clip role_history effective_to at exit date
    def clip_rh(row):
        ed = exit_dates.get(row["employee_id"])
        if ed is None:
            return row
        if row["effective_from"] > ed:
            return None
        if row["effective_to"] is None or row["effective_to"] > ed:
            row = row.copy()
            row["effective_to"] = ed
        return row

    role_history = role_history.apply(clip_rh, axis=1).dropna()

    # Re-cast integer columns — apply() with None can promote int->float
    int_cols = {
        "employees":            ["employee_id"],
        "role_history":         ["employee_id", "role_level", "salary"],
        "performance_reviews":  ["review_id", "employee_id", "rating"],
        "absence_events":       ["absence_id", "employee_id", "days"],
        "exits":                ["exit_id", "employee_id"],
    }
    for df, cols in zip(
        [employees, role_history, reviews, absences, exits],
        int_cols.values(),
    ):
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

    # Drop individual_risk — it's a latent variable, not observable
    employees_out = employees.drop(columns=["individual_risk"])

    employees_out.to_csv(OUTPUT_DIR / "employees.csv", index=False)
    role_history.to_csv(OUTPUT_DIR / "role_history.csv", index=False)
    reviews.to_csv(OUTPUT_DIR / "performance_reviews.csv", index=False)
    absences.to_csv(OUTPUT_DIR / "absence_events.csv", index=False)
    exits.to_csv(OUTPUT_DIR / "exits.csv", index=False)

    emp_hire_map = employees.set_index("employee_id")["hire_date"].to_dict()
    total_active_years = sum(
        max((min(exit_dates.get(e, END_DATE), END_DATE) - max(emp_hire_map[e], START_DATE)).days, 0)
        for e in employees["employee_id"]
    ) / 365.25

    print("\n--- Output summary ---")
    print(f"employees:           {len(employees_out):>6,}")
    print(f"role_history rows:   {len(role_history):>6,}")
    print(f"performance_reviews: {len(reviews):>6,}")
    print(f"absence_events:      {len(absences):>6,}")
    print(f"exits:               {len(exits):>6,}")
    attrition_rate = len(exits) / total_active_years if total_active_years > 0 else 0
    print(f"annualised attrition: {attrition_rate:.1%}")

    risks = employees["individual_risk"].values
    print(f"\nindividual_risk: mean={risks.mean():.3f}  p90={np.percentile(risks, 90):.3f}  max={risks.max():.3f}")
    print(f"CSVs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
