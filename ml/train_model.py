"""
Train an attrition risk model on fct_attrition_snapshots.

Split: train on snapshots <= 2024-12-31, test on 2025 snapshots.
No employee-level leakage: the label is forward-looking per snapshot,
and the split is on snapshot_date (not employee_id), so the same
employee can appear in both sets at different points in time — that is
correct, not leakage, because the features differ.

If AUC > 0.95, investigate for leakage before celebrating.

Usage:
    python ml/train_model.py [--project YOUR_PROJECT] [--dataset peopleops_prod]
"""

import argparse
import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "age_at_snapshot",
    "tenure_months",
    "role_level",
    "salary_vs_median_pct",
    "months_in_current_role",
    "avg_rating_all",           # all historical ratings — low noise, primary signal
    "absence_events_last90",
    "absence_events_per_month", # lifetime rate — removes tenure-length bias
]

CATEGORICAL_FEATURES = [
    "department",
    "contract_type",
]

# Gender excluded from model features — used for fairness audit only,
# not as a predictor (avoids disparate impact).
FAIRNESS_COLS = ["gender"]

TARGET = "exited_within_6m"
TRAIN_CUTOFF = "2024-12-31"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT"))
    parser.add_argument("--dataset", default="peopleops_prod")
    return parser.parse_args()


def load_snapshots(project: str, dataset: str) -> pd.DataFrame:
    client = bigquery.Client(project=project)
    query = f"""
        select
            employee_id,
            snapshot_date,
            gender,
            department,
            contract_type,
            age_at_snapshot,
            tenure_months,
            role_level,
            salary_vs_median_pct,
            months_in_current_role,
            avg_rating_all,
            absence_events_last90,
            absence_events_per_month,
            exited_within_6m
        from `{project}.{dataset}.fct_attrition_snapshots`
        order by snapshot_date, employee_id
    """
    print(f"Reading fct_attrition_snapshots from {project}.{dataset}...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df):,} rows, {df['exited_within_6m'].mean():.1%} positive rate")
    return df


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Precision among the top-k% of scores — the realistic HR use case."""
    n = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_prob)[::-1][:n]
    return y_true[top_idx].mean()


def train(df: pd.DataFrame) -> None:
    # Time-based split — cast cutoff to date to match BigQuery date column
    import datetime
    cutoff = datetime.date.fromisoformat(TRAIN_CUTOFF)
    train_df = df[df["snapshot_date"] <= cutoff].copy()
    test_df  = df[df["snapshot_date"] >  cutoff].copy()

    print(f"\nTrain: {len(train_df):,} rows ({train_df['exited_within_6m'].mean():.1%} positive)")
    print(f"Test:  {len(test_df):,} rows  ({test_df['exited_within_6m'].mean():.1%} positive)")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET].values
    X_test  = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test  = test_df[TARGET].values

    # Preprocessing + model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=1.0,
        )),
    ])

    print("\nFitting model...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test  = pipeline.predict_proba(X_test)[:, 1]

    roc_train = roc_auc_score(y_train, y_prob_train)
    roc_test  = roc_auc_score(y_test,  y_prob_test)
    pr_train  = average_precision_score(y_train, y_prob_train)
    pr_test   = average_precision_score(y_test,  y_prob_test)
    p_at_10_test = precision_at_top_k(y_test, y_prob_test, k=0.10)

    print("\n--- Evaluation ---")
    print(f"  ROC-AUC      train={roc_train:.3f}  test={roc_test:.3f}")
    print(f"  PR-AUC       train={pr_train:.3f}  test={pr_test:.3f}")
    print(f"  Precision@10%              test={p_at_10_test:.3f}")

    if roc_test > 0.95:
        print("\n  WARNING: AUC > 0.95 — investigate for leakage before trusting this model.")

    print("\n--- Classification report (test, threshold 0.5) ---")
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_test, target_names=["stayed", "exited"]))

    # Feature names after one-hot encoding
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    lr = pipeline.named_steps["classifier"]

    # Feature importance via LR coefficients (equivalent to SHAP for linear models)
    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    mean_abs_contrib = np.abs(lr.coef_[0] * X_test_transformed).mean(axis=0)
    importance = pd.Series(mean_abs_contrib, index=all_feature_names).sort_values(ascending=False)

    print("\n--- Top 10 features by mean |SHAP| ---")
    for feat, imp in importance.head(10).items():
        coef = lr.coef_[0][all_feature_names.index(feat)]
        direction = "^ risk" if coef > 0 else "v risk"
        print(f"  {feat:<35} {imp:.4f}  ({direction})")

    # Odds ratios for numeric features (interpretable for stakeholders)
    print("\n--- Odds ratios (numeric features, 1-SD change) ---")
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    for i, feat in enumerate(NUMERIC_FEATURES):
        coef = lr.coef_[0][i]
        odds_ratio = np.exp(coef)
        print(f"  {feat:<35} OR={odds_ratio:.2f}")

    # Fairness audit: score gap by gender
    test_df = test_df.copy()
    test_df["risk_score"] = y_prob_test
    print("\n--- Fairness audit: mean risk score by gender ---")
    print(test_df.groupby("gender")["risk_score"].agg(["mean", "median", "count"]).round(3))

    # Save artifacts
    metrics = {
        "roc_auc_train": round(roc_train, 4),
        "roc_auc_test":  round(roc_test,  4),
        "pr_auc_train":  round(pr_train,  4),
        "pr_auc_test":   round(pr_test,   4),
        "precision_at_top10_test": round(float(p_at_10_test), 4),
        "train_rows": int(len(train_df)),
        "test_rows":  int(len(test_df)),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate":  round(float(y_test.mean()),  4),
        "top5_features": importance.head(5).index.tolist(),
        "train_cutoff": TRAIN_CUTOFF,
    }

    joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.joblib")
    joblib.dump({"feature_names": all_feature_names, "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
                ARTIFACTS_DIR / "feature_meta.joblib")

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR.resolve()}")
    print(f"  pipeline.joblib  |  feature_meta.joblib  |  metrics.json")


def main() -> None:
    args = get_args()
    if not args.project:
        raise SystemExit("Set GCP_PROJECT env var or pass --project")

    df = load_snapshots(args.project, args.dataset)
    train(df)


if __name__ == "__main__":
    main()
