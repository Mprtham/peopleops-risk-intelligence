"""
Attrition Risk API

POST /predict   {"employee_id": int}
    → fetch the employee's most recent snapshot from BigQuery,
      score it, explain with SHAP, return risk score + top factors.

GET  /health    → liveness check + cached model metrics

Run:
    uvicorn ml.risk_api:app --reload
    # or from repo root:
    make api
"""

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import bigquery
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

NUMERIC_FEATURES = [
    "age_at_snapshot",
    "tenure_months",
    "role_level",
    "salary_vs_median_pct",
    "months_in_current_role",
    "avg_rating_all",
    "absence_events_last90",
    "absence_events_per_month",
]
CATEGORICAL_FEATURES = ["department", "contract_type"]
ALL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
BQ_DATASET  = os.environ.get("BQ_DATASET", "peopleops_prod")


# ---------------------------------------------------------------------------
# Artifact loading (once at startup)
# ---------------------------------------------------------------------------

class ModelState:
    pipeline = None
    explainer = None
    feature_meta: dict = {}
    metrics: dict = {}
    bq_client: bigquery.Client | None = None


state = ModelState()


def load_artifacts() -> None:
    required = ["pipeline.joblib", "shap_explainer.joblib", "feature_meta.joblib", "metrics.json"]
    missing = [f for f in required if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing artifacts: {missing}. Run `python ml/train_model.py` first."
        )

    state.pipeline     = joblib.load(ARTIFACTS_DIR / "pipeline.joblib")
    state.explainer    = joblib.load(ARTIFACTS_DIR / "shap_explainer.joblib")
    state.feature_meta = joblib.load(ARTIFACTS_DIR / "feature_meta.joblib")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        state.metrics = json.load(f)


def get_bq_client() -> bigquery.Client:
    if state.bq_client is None:
        if not GCP_PROJECT:
            raise RuntimeError("Set GCP_PROJECT environment variable.")
        state.bq_client = bigquery.Client(project=GCP_PROJECT)
    return state.bq_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    # Pre-warm BigQuery client so the first request isn't slow
    if GCP_PROJECT:
        get_bq_client()
    yield
    # Cleanup (none needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PeopleOps Attrition Risk API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    employee_id: int = Field(..., description="Employee ID to score")


class FactorDetail(BaseModel):
    feature: str
    shap_effect: float
    raw_value: Any
    direction: str   # "increases_risk" | "decreases_risk"


class PredictResponse(BaseModel):
    employee_id: int
    snapshot_date: str
    risk_score: float
    risk_tier: str          # "high" | "medium" | "low"
    top_factors: list[FactorDetail]
    playbook_action: str


# ---------------------------------------------------------------------------
# BigQuery: fetch latest snapshot for one employee
# ---------------------------------------------------------------------------

SNAPSHOT_QUERY = """
    select
        employee_id,
        snapshot_date,
        department,
        contract_type,
        age_at_snapshot,
        tenure_months,
        role_level,
        salary_vs_median_pct,
        months_in_current_role,
        avg_rating_all,
        absence_events_last90,
        absence_events_per_month
    from `{project}.{dataset}.fct_attrition_snapshots`
    where employee_id = {employee_id}
    order by snapshot_date desc
    limit 1
"""


def fetch_latest_snapshot(employee_id: int) -> pd.DataFrame:
    client = get_bq_client()
    query = SNAPSHOT_QUERY.format(
        project=GCP_PROJECT,
        dataset=BQ_DATASET,
        employee_id=employee_id,
    )
    df = client.query(query).to_dataframe()
    return df


# ---------------------------------------------------------------------------
# Scoring + explanation
# ---------------------------------------------------------------------------

def risk_tier(score: float) -> str:
    if score > 0.70:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def playbook_action(score: float, row: pd.Series) -> str:
    if score > 0.70:
        return "Stay interview within 5 working days (HRBP owner)"
    if score >= 0.50 and row.get("months_in_current_role", 0) >= 24:
        return "Flag for promotion committee at next cycle (HRBP + line manager)"
    if score >= 0.50 and row.get("absence_events_per_month", 0) > 1.0:
        return "Workload review + wellbeing check-in within 5 working days (line manager)"
    if score >= 0.40 and row.get("salary_vs_median_pct", 0) < -0.10:
        return "Compensation review against band at next comp cycle"
    if score >= 0.40 and row.get("avg_rating_all", 3) < 2.5:
        return "Performance support plan within 10 working days (line manager)"
    return "Monitor — schedule check-in at next 1:1"


def score_employee(employee_id: int) -> PredictResponse:
    df = fetch_latest_snapshot(employee_id)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot found for employee_id={employee_id}. "
                   "Employee may not exist or may already have exited.",
        )

    row = df.iloc[0]
    X = df[ALL_INPUT_FEATURES]

    # Predict
    risk_score = float(state.pipeline.predict_proba(X)[0, 1])

    # SHAP explanation
    X_transformed = state.pipeline.named_steps["preprocessor"].transform(X)
    shap_vals = state.explainer(X_transformed)

    # Map SHAP values back to all feature names (numeric + OHE-expanded categoricals)
    all_feature_names: list[str] = state.feature_meta["feature_names"]
    shap_effects = shap_vals.values[0]  # shape: (n_features,)

    # For display, collapse OHE features back to their parent categorical name
    # by summing SHAP effects within each categorical group
    collapsed: dict[str, float] = {}
    for feat, effect in zip(all_feature_names, shap_effects):
        parent = feat.split("_")[0] if feat not in NUMERIC_FEATURES else feat
        # Keep numeric as-is; collapse OHE back to base name
        display_name = feat if feat in NUMERIC_FEATURES else feat
        collapsed[feat] = effect

    # Sort by abs effect, take top 5
    sorted_feats = sorted(collapsed.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    top_factors = []
    for feat_name, effect in sorted_feats:
        # Get the raw value for this feature (use parent name for categoricals)
        if feat_name in NUMERIC_FEATURES:
            raw_val = float(row[feat_name]) if feat_name in row.index else None
        else:
            # OHE feature — find parent column
            parent_col = next(
                (c for c in CATEGORICAL_FEATURES if feat_name.startswith(c + "_")),
                None,
            )
            raw_val = str(row[parent_col]) if parent_col and parent_col in row.index else None

        top_factors.append(FactorDetail(
            feature=feat_name,
            shap_effect=round(float(effect), 4),
            raw_value=raw_val,
            direction="increases_risk" if effect > 0 else "decreases_risk",
        ))

    return PredictResponse(
        employee_id=int(row["employee_id"]),
        snapshot_date=str(row["snapshot_date"]),
        risk_score=round(risk_score, 4),
        risk_tier=risk_tier(risk_score),
        top_factors=top_factors,
        playbook_action=playbook_action(risk_score, row),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return score_employee(request.employee_id)


@app.get("/health")
def health() -> JSONResponse:
    artifacts_ok = all(
        (ARTIFACTS_DIR / f).exists()
        for f in ["pipeline.joblib", "shap_explainer.joblib"]
    )
    return JSONResponse({
        "status": "ok" if artifacts_ok else "degraded",
        "artifacts_loaded": state.pipeline is not None,
        "gcp_project": GCP_PROJECT or "(not set)",
        "bq_dataset": BQ_DATASET,
        "model_metrics": state.metrics,
    })
