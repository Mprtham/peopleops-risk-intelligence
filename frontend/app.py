"""
PeopleOps Risk Intelligence - Streamlit frontend.
Polished commercial SaaS design. All model logic, queries, and calculations unchanged.
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery

warnings.filterwarnings("ignore")


def _init_gcp_auth() -> None:
    """
    On Streamlit Cloud, load the service account from st.secrets and write it
    to a temp file so google-cloud-bigquery picks it up automatically.
    On local dev, gcloud ADC handles auth — this is a no-op.
    """
    if "gcp_service_account" not in st.secrets:
        return
    key = dict(st.secrets["gcp_service_account"])
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(key, tmp)
    tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


_init_gcp_auth()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PeopleOps Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system - single CSS injection
# ---------------------------------------------------------------------------
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary:      #1F4E5F;
  --primary-dark: #163845;
  --accent-red:   #E5484D;
  --accent-amber: #F5A623;
  --accent-green: #30A46C;
  --bg:           #F7F8FA;
  --card-bg:      #FFFFFF;
  --text:         #1A1D1F;
  --muted:        #6F767E;
  --border:       #E8EAED;
  --shadow:       0 1px 3px rgba(0,0,0,0.08);
  --radius:       12px;
  --radius-sm:    8px;
  --radius-xs:    6px;
}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text);
}

/* --- Hide default Streamlit chrome --- */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
  visibility: hidden !important;
}

/* --- Layout --- */
.stApp { background-color: var(--bg); }

.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 5rem !important;
  max-width: 1280px;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
  background-color: var(--card-bg);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem;
}

/* --- Metric cards (sidebar) --- */
.mc {
  background: var(--card-bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  padding: 16px 20px;
  margin-bottom: 10px;
}
.mc-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.mc-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  display: inline-block;
}
.mc-value {
  font-size: 28px;
  font-weight: 600;
  line-height: 1.1;
  color: var(--text);
}
.mc-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 3px;
}

/* --- Section header in sidebar --- */
.sb-section {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 16px 0 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 12px;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
  gap: 24px;
  border-bottom: 1.5px solid var(--border);
  background: transparent;
  padding: 0;
}
.stTabs [data-baseweb="tab"] {
  font-size: 15px;
  font-weight: 500;
  color: var(--muted);
  padding: 10px 0;
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1.5px;
  transition: color 0.15s ease;
}
.stTabs [aria-selected="true"] {
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--primary) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 24px;
  animation: fadein 200ms ease-out;
}
@keyframes fadein {
  from { opacity: 0; transform: translateY(3px); }
  to   { opacity: 1; transform: translateY(0);   }
}

/* --- Card --- */
.card {
  background: var(--card-bg);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
  padding: 24px;
  margin-bottom: 16px;
}

/* --- Risk pill --- */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.02em;
  line-height: 1;
}
.pill.high   { background: rgba(229,72,77,0.12);  color: #B91C22; }
.pill.medium { background: rgba(245,166,35,0.12); color: #996800; }
.pill.low    { background: rgba(48,164,108,0.12); color: #1A7A4A; }

/* --- Definition grid (employee details) --- */
.def-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px 32px;
}
.def-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
.def-value {
  font-size: 15px;
  font-weight: 500;
  color: var(--text);
}

/* --- Playbook action box --- */
.pb-box {
  border-radius: var(--radius-sm);
  padding: 16px 20px;
  margin-top: 16px;
  border-left: 4px solid;
  display: flex;
  gap: 14px;
  align-items: flex-start;
}
.pb-box.high   { border-color: var(--accent-red);   background: rgba(229,72,77,0.05);  }
.pb-box.medium { border-color: var(--accent-amber); background: rgba(245,166,35,0.05); }
.pb-box.low    { border-color: var(--accent-green); background: rgba(48,164,108,0.05); }
.pb-icon {
  font-size: 18px;
  line-height: 1;
  flex-shrink: 0;
  margin-top: 1px;
}
.pb-action {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}
.pb-owner {
  font-size: 13px;
  color: var(--muted);
}

/* --- Primary button --- */
.stButton > button[kind="primary"] {
  background-color: var(--primary) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 0.5rem 1.25rem !important;
  transition: background-color 0.15s ease !important;
  box-shadow: none !important;
}
.stButton > button[kind="primary"]:hover {
  background-color: var(--primary-dark) !important;
}
.stButton > button[kind="secondary"] {
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
}

/* --- Number input / selectbox --- */
.stNumberInput input,
.stSelectbox > div > div {
  border-radius: var(--radius-sm) !important;
  font-family: 'Inter', sans-serif !important;
}

/* --- Hero header --- */
.hero-title {
  font-size: 34px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.5px;
  line-height: 1.15;
  margin: 0 0 10px 0;
}
.hero-subtitle {
  font-size: 15px;
  color: var(--muted);
  line-height: 1.6;
  max-width: 820px;
  margin: 0 0 16px 0;
}

/* --- Pipeline chips --- */
.chip-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 24px;
}
.chip {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-xs);
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  white-space: nowrap;
}
.chip-arrow {
  color: var(--muted);
  font-size: 14px;
  line-height: 1;
}

/* --- Leaderboard HTML table --- */
.lb-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--card-bg);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.lb-table th {
  background: #F7F8FA;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--muted);
  padding: 12px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.lb-table td {
  padding: 11px 14px;
  border-bottom: 1px solid #F2F3F5;
  color: var(--text);
  vertical-align: middle;
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr:hover td { background: #F7F8FA; }
.lb-score { font-weight: 700; font-size: 15px; }

/* --- Info callout --- */
.callout {
  background: #EEF6FB;
  border-left: 4px solid var(--primary);
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  font-size: 13px;
  color: var(--text);
  margin-top: 16px;
  line-height: 1.6;
}

/* --- Section divider --- */
.section-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 24px 0 12px;
}
.section-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
  margin: 24px 0 4px;
}

/* --- Fixed footer --- */
.app-footer {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 44px;
  background: var(--bg);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  color: var(--muted);
  z-index: 9999;
}
.app-footer a {
  color: var(--primary);
  text-decoration: none;
  font-weight: 500;
}
.app-footer a:hover { text-decoration: underline; }

/* --- Mobile responsive --- */
@media (max-width: 768px) {
  .def-grid { grid-template-columns: 1fr; }
  .chip-row { gap: 4px; }
  .hero-title { font-size: 26px; }
}
</style>
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FONT = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

C = {
    "primary": "#1F4E5F",
    "red":     "#E5484D",
    "amber":   "#F5A623",
    "green":   "#30A46C",
    "muted":   "#6F767E",
    "border":  "#E8EAED",
    "bg":      "rgba(0,0,0,0)",
    "text":    "#1A1D1F",
}

TIER_COLOR = {"high": C["red"], "medium": C["amber"], "low": C["green"]}
TIER_LABEL = {"high": "HIGH RISK", "medium": "MEDIUM RISK", "low": "LOW RISK"}
TIER_ICON  = {"high": "⚠", "medium": "○", "low": "✓"}  # unicode: warning, circle, check

ARTIFACTS_DIR = Path(__file__).parent.parent / "ml" / "artifacts"
GCP_PROJECT   = os.environ.get("GCP_PROJECT", "")
BQ_DATASET    = os.environ.get("BQ_DATASET", "peopleops_dev_marts")

NUMERIC_FEATURES = [
    "age_at_snapshot", "tenure_months", "role_level", "salary_vs_median_pct",
    "months_in_current_role", "avg_rating_all",
    "absence_events_last90", "absence_events_per_month",
]
CATEGORICAL_FEATURES = ["department", "contract_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS = {
    "avg_rating_all":           "Avg performance rating (all-time)",
    "absence_events_per_month": "Absence rate (events/month)",
    "absence_events_last90":    "Absence events (last 90 days)",
    "months_in_current_role":   "Months in current role",
    "tenure_months":            "Tenure (months)",
    "role_level":               "Role level (0=entry, 4=director)",
    "salary_vs_median_pct":     "Salary vs band median (%)",
    "age_at_snapshot":          "Age",
    "department":               "Department",
    "contract_type":            "Contract type",
}

# ---------------------------------------------------------------------------
# Inject design system
# ---------------------------------------------------------------------------
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caching (logic unchanged)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML artifacts...")
def load_artifacts():
    # sklearn 1.6+ removed _RemainderColsList; inject a stand-in so artifacts
    # pickled with 1.5.x can be loaded by newer versions without retraining.
    import sklearn.compose._column_transformer as _sct
    if not hasattr(_sct, "_RemainderColsList"):
        _sct._RemainderColsList = list

    pipeline     = joblib.load(ARTIFACTS_DIR / "pipeline.joblib")
    feature_meta = joblib.load(ARTIFACTS_DIR / "feature_meta.joblib")
    metrics      = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    return pipeline, feature_meta, metrics


@st.cache_resource(show_spinner="Connecting to BigQuery...")
def get_bq_client(project: str):
    return bigquery.Client(project=project)


@st.cache_data(ttl=300, show_spinner="Fetching snapshot data...")
def load_all_snapshots(project: str, dataset: str) -> pd.DataFrame:
    client = get_bq_client(project)
    return client.query(f"""
        select
            employee_id, snapshot_date, gender, department, contract_type,
            age_at_snapshot, tenure_months, role_level, salary_vs_median_pct,
            months_in_current_role, avg_rating_all,
            absence_events_last90, absence_events_per_month,
            exited_within_6m
        from `{project}.{dataset}.fct_attrition_snapshots`
        where snapshot_date >= date_sub(current_date(), interval 2 year)
        order by snapshot_date desc, employee_id
    """).to_dataframe()


# ---------------------------------------------------------------------------
# Model helpers (logic unchanged)
# ---------------------------------------------------------------------------

def risk_tier(score: float) -> str:
    if score > 0.70: return "high"
    if score >= 0.40: return "medium"
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
    return "Monitor - schedule check-in at next 1:1"


def score_dataframe(df: pd.DataFrame, pipeline, feature_meta) -> pd.DataFrame:
    X = df[ALL_FEATURES]
    df = df.copy()
    df["risk_score"] = pipeline.predict_proba(X)[:, 1]
    df["risk_tier"]  = df["risk_score"].apply(risk_tier)
    return df


def feature_contributions_for_row(row: pd.Series, pipeline, feature_meta) -> tuple:
    X   = pd.DataFrame([row[ALL_FEATURES]])
    X_t = pipeline.named_steps["preprocessor"].transform(X)
    lr  = pipeline.named_steps["classifier"]
    contributions = lr.coef_[0] * X_t[0]
    base = float(1 / (1 + np.exp(-float(lr.intercept_[0]))))
    return feature_meta["feature_names"], contributions, base


def parse_playbook(text: str) -> tuple[str, str]:
    """Split 'Do X (Owner)' into (action, owner). Logic unchanged."""
    if "(" in text and text.endswith(")"):
        action = text[:text.rfind("(")].strip()
        owner  = text[text.rfind("(")+1:-1]
        return action, owner
    return text, ""


# ---------------------------------------------------------------------------
# HTML component builders
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = "", dot_color: str = "") -> str:
    dot = (
        f'<span class="mc-dot" style="background:{dot_color}"></span>'
        if dot_color else ""
    )
    sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="mc">
      <div class="mc-label">{dot}{label}</div>
      <div class="mc-value">{value}</div>
      {sub_html}
    </div>
    """


def risk_pill(tier: str) -> str:
    label = TIER_LABEL[tier]
    return f'<span class="pill {tier}">{label}</span>'


def def_grid(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div><div class="def-label">{label}</div><div class="def-value">{val}</div></div>'
        for label, val in items
    )
    return f'<div class="def-grid">{cells}</div>'


def playbook_box_html(tier: str, action: str, owner: str) -> str:
    icon = TIER_ICON[tier]
    owner_html = f'<div class="pb-owner">Owner: {owner}</div>' if owner else ""
    return f"""
    <div class="pb-box {tier}">
      <div class="pb-icon">{icon}</div>
      <div>
        <div class="pb-action">{action}</div>
        {owner_html}
      </div>
    </div>
    """


def leaderboard_html(df: pd.DataFrame) -> str:
    rows = ""
    for _, r in df.iterrows():
        tier  = r["risk_tier"]
        pill  = risk_pill(tier)
        score = f"{r['risk_score']:.1%}"
        rows += f"""
        <tr>
          <td><strong>{int(r['employee_id'])}</strong></td>
          <td>{r['department']}</td>
          <td>{r['contract_type']}</td>
          <td>{int(r['tenure_months'])}</td>
          <td>{int(r['role_level'])}</td>
          <td>{r['avg_rating_all']:.2f}</td>
          <td>{r['absence_events_per_month']:.2f}</td>
          <td class="lb-score">{score}</td>
          <td>{pill}</td>
        </tr>"""
    return f"""
    <table class="lb-table">
      <thead><tr>
        <th>ID</th><th>Department</th><th>Contract</th>
        <th>Tenure (mo)</th><th>Level</th><th>Avg Rating</th>
        <th>Absences/mo</th><th>Risk Score</th><th>Tier</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Plotly charts (restyled, logic unchanged)
# ---------------------------------------------------------------------------

PLOTLY_BASE = dict(
    font=dict(family=FONT, color=C["text"]),
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["bg"],
    margin=dict(l=10, r=10, t=30, b=10),
)


def gauge_chart(score: float) -> go.Figure:
    tier   = risk_tier(score)
    colour = TIER_COLOR[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={
            "suffix": "%",
            "valueformat": ".1f",
            "font": {"size": 52, "color": colour, "family": FONT},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": C["border"],
                "tickfont": {"family": FONT, "size": 11, "color": C["muted"]},
            },
            "bar":  {"color": "#2D3748", "thickness": 0.18},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(48,164,108,0.10)"},
                {"range": [40, 70], "color": "rgba(245,166,35,0.10)"},
                {"range": [70, 100],"color": "rgba(229,72,77,0.10)"},
            ],
        },
        title={
            "text": "6-month exit probability",
            "font": {"size": 13, "color": C["muted"], "family": FONT},
        },
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        height=240,
        margin=dict(t=48, b=8, l=16, r=16),
    )
    return fig


def shap_waterfall(feature_names, shap_values, base_value, n: int = 8) -> go.Figure:
    idx        = np.argsort(np.abs(shap_values))[::-1][:n]
    sel_names  = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in idx]
    sel_values = [shap_values[i] for i in idx]
    colours    = [C["red"] if v > 0 else C["green"] for v in sel_values]

    fig = go.Figure(go.Bar(
        x=sel_values[::-1],
        y=sel_names[::-1],
        orientation="h",
        marker_color=colours[::-1],
        marker_line_width=0,
        text=[f"{'+' if v>0 else ''}{v:.3f}" for v in sel_values[::-1]],
        textposition="outside",
        textfont={"family": FONT, "size": 12, "color": C["text"]},
        hovertemplate="%{y}<br>SHAP effect: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title={"text": "What drove this score? (SHAP values)", "font": {"size": 14, "family": FONT}},
        xaxis=dict(
            title=dict(
                text="Effect on risk score    ← lowers risk         raises risk →",
                font={"size": 11, "color": C["muted"], "family": FONT},
            ),
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor=C["muted"],
            showgrid=False,
        ),
        yaxis=dict(showgrid=False),
        height=300,
        margin=dict(t=40, b=16, l=10, r=80),
    )
    return fig


def history_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["snapshot_date"].astype(str),
        y=hist["risk_score"],
        mode="lines",
        line=dict(color=C["primary"], width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(31,78,95,0.08)",
        name="Risk score",
        hovertemplate="%{x}<br>Risk: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(
        y=0.70, line_dash="dash", line_color=C["red"], line_width=1,
        annotation_text="High risk", annotation_position="right",
        annotation_font={"color": C["red"], "size": 11, "family": FONT},
    )
    fig.add_hline(
        y=0.40, line_dash="dash", line_color=C["amber"], line_width=1,
        annotation_text="Medium risk", annotation_position="right",
        annotation_font={"color": C["amber"], "size": 11, "family": FONT},
    )
    fig.update_layout(
        **PLOTLY_BASE,
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor=C["border"],
            gridwidth=1,
            zeroline=False,
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        hovermode="x unified",
        height=220,
        margin=dict(t=30, b=10, l=10, r=90),
        showlegend=False,
    )
    return fig


def dept_bar_chart(latest_df: pd.DataFrame) -> go.Figure:
    agg = latest_df.groupby("department").agg(
        avg_risk  =("risk_score", "mean"),
        headcount =("employee_id", "count"),
        high_risk =("risk_tier", lambda x: (x == "high").sum()),
    ).reset_index().sort_values("avg_risk", ascending=True)

    colours = [TIER_COLOR[risk_tier(s)] for s in agg["avg_risk"]]

    fig = go.Figure(go.Bar(
        x=agg["avg_risk"],
        y=agg["department"],
        orientation="h",
        marker_color=colours,
        marker_line_width=0,
        text=[f"{s:.1%}" for s in agg["avg_risk"]],
        textposition="outside",
        textfont={"family": FONT, "size": 12},
        customdata=np.stack([
            agg["headcount"],
            agg["high_risk"],
            (agg["avg_risk"] * 100).round(1),
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg risk: %{customdata[2]:.1f}%<br>"
            "Headcount: %{customdata[0]}<br>"
            "High-risk employees: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title={"text": "Average predicted risk by department", "font": {"size": 14, "family": FONT}},
        xaxis=dict(tickformat=".0%", range=[0, 0.9], showgrid=False),
        yaxis=dict(showgrid=False),
        height=300,
        margin=dict(t=40, b=10, l=10, r=80),
    )
    return fig


def dept_stack_chart(latest_df: pd.DataFrame) -> go.Figure:
    tier_counts = (
        latest_df.groupby(["department", "risk_tier"])
        .size().reset_index(name="count")
    )
    pivot = tier_counts.pivot(index="department", columns="risk_tier", values="count").fillna(0)
    for col in ["high", "medium", "low"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["high", "medium", "low"]].sort_values("high", ascending=False)

    fig = go.Figure()
    tier_cfg = [
        ("high",   C["red"],   "High risk"),
        ("medium", C["amber"], "Medium risk"),
        ("low",    C["green"], "Low risk"),
    ]
    for key, colour, name in tier_cfg:
        fig.add_trace(go.Bar(
            name=name,
            x=pivot.index,
            y=pivot[key],
            marker_color=colour,
            marker_line_width=0,
            hovertemplate=f"{name}: %{{y}}<br>Department: %{{x}}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_BASE,
        barmode="stack",
        title={"text": "Risk tier breakdown by department", "font": {"size": 14, "family": FONT}},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=C["border"]),
        height=320,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font={"family": FONT, "size": 13},
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Session state for leaderboard -> profile jump
    if "emp_id_input" not in st.session_state:
        st.session_state["emp_id_input"] = None  # filled after data loads

    # ----- GCP project -----
    project = GCP_PROJECT
    if not project:
        project = st.text_input(
            "GCP Project ID",
            placeholder="peopleops-analytics",
            help="Set GCP_PROJECT env var to skip this prompt",
        )
        if not project:
            st.info("Enter your GCP project ID above to connect to BigQuery.")
            st.stop()

    # ----- Load data -----
    pipeline, feature_meta, metrics = load_artifacts()
    all_df    = load_all_snapshots(project, BQ_DATASET)
    scored_df = score_dataframe(all_df, pipeline, feature_meta)

    latest_date = scored_df["snapshot_date"].max()
    latest_df   = scored_df[scored_df["snapshot_date"] == latest_date]

    min_id      = int(scored_df["employee_id"].min())
    max_id      = int(scored_df["employee_id"].max())
    default_id  = int(scored_df[scored_df["risk_tier"] == "high"]["employee_id"].iloc[0])

    # Initialise session state default after data is available
    if st.session_state["emp_id_input"] is None:
        st.session_state["emp_id_input"] = default_id

    # ----- Sidebar -----
    with st.sidebar:
        n_active = len(latest_df)
        n_high   = int((latest_df["risk_tier"] == "high").sum())
        n_medium = int((latest_df["risk_tier"] == "medium").sum())
        n_low    = int((latest_df["risk_tier"] == "low").sum())
        top_dept = latest_df.groupby("department")["risk_score"].mean().idxmax()
        pct_high = n_high / n_active if n_active else 0

        st.markdown('<div class="sb-section">Company Overview</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:12px;color:#6F767E;margin:-4px 0 12px">Snapshot: {latest_date}</p>', unsafe_allow_html=True)

        st.markdown(
            metric_card("Active employees", f"{n_active:,}", f"as of {latest_date}"),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("High-risk employees", f"{n_high:,}", f"{pct_high:.1%} of workforce", dot_color=C["red"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("Medium-risk employees", f"{n_medium:,}", dot_color=C["amber"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("Highest-risk department", top_dept),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section">Model Performance</div>', unsafe_allow_html=True)
        st.markdown(
            metric_card("ROC-AUC (test 2025)", f"{metrics['roc_auc_test']:.3f}", "time-based split"),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("PR-AUC (test 2025)", f"{metrics['pr_auc_test']:.3f}", "vs 0.156 random baseline"),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("Precision @ top 10%", f"{metrics['precision_at_top10_test']:.1%}", "of flagged employees leave"),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section">How It Works</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:13px;color:#6F767E;line-height:1.7;margin:0">'
            '1. BigQuery stores 5 raw HR event tables<br>'
            '2. dbt builds leakage-proof snapshots, features computed as of each quarter-end<br>'
            '3. Logistic regression + SHAP trained on 2023-2024 data<br>'
            '4. This app scores live data and explains every prediction'
            '</p>',
            unsafe_allow_html=True,
        )

    # ----- Hero header -----
    st.markdown(
        '<h1 class="hero-title">PeopleOps Risk Intelligence</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle">'
        'PeopleOps Risk Intelligence is an end to end HR analytics platform that predicts which employees are likely to '
        'leave within the next six months and tells the business what to do about it. Synthetic HR data flows through '
        'BigQuery and dbt into a tested warehouse, an interpretable logistic regression model scores every employee at '
        'each quarterly snapshot, SHAP explains the drivers behind each score, and a retention playbook converts every '
        'prediction into a concrete action with an owner and a deadline.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="chip-row">'
        '<span class="chip">BigQuery</span>'
        '<span class="chip-arrow">&rarr;</span>'
        '<span class="chip">dbt</span>'
        '<span class="chip-arrow">&rarr;</span>'
        '<span class="chip">Logistic Regression + SHAP</span>'
        '<span class="chip-arrow">&rarr;</span>'
        '<span class="chip">Streamlit</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ----- Tabs -----
    tab1, tab2, tab3 = st.tabs(
        ["Employee Risk Profile", "At-Risk Leaderboard", "Department Overview"]
    )

    # ================================================================
    # TAB 1 - Employee Risk Profile
    # ================================================================
    with tab1:
        col_in, col_btn = st.columns([4, 1], vertical_alignment="bottom")
        with col_in:
            emp_id = st.number_input(
                "Employee ID",
                min_value=min_id,
                max_value=max_id,
                step=1,
                key="emp_id_input",
                help="Enter any employee ID, or select one from the At-Risk Leaderboard tab",
            )
        with col_btn:
            analyse = st.button("Analyse", type="primary", use_container_width=True)

        emp_rows = scored_df[scored_df["employee_id"] == emp_id]
        if emp_rows.empty:
            st.warning(f"No snapshots found for employee {emp_id}.")
            st.stop()

        row    = emp_rows.sort_values("snapshot_date").iloc[-1]
        score  = float(row["risk_score"])
        tier   = row["risk_tier"]

        with st.spinner("Scoring employee..."):
            action_full     = playbook_action(score, row)
            action, owner   = parse_playbook(action_full)
            names, shap_vals, base = feature_contributions_for_row(row, pipeline, feature_meta)

        st.toast(f"Profile loaded for employee {int(emp_id)}")

        # Employee header with pill badge
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin:16px 0 20px">'
            f'<span style="font-size:22px;font-weight:700;color:#1A1D1F">Employee {int(emp_id)}</span>'
            f'{risk_pill(tier)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        c_gauge, c_details = st.columns([1, 1], gap="large")

        with c_gauge:
            st.plotly_chart(
                gauge_chart(score),
                use_container_width=False,
                config={"displayModeBar": False},
            )

        with c_details:
            st.markdown(
                '<p class="section-label">Employee Details</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                def_grid([
                    ("Department",          row["department"]),
                    ("Contract type",       row["contract_type"]),
                    ("Tenure",              f"{int(row['tenure_months'])} months"),
                    ("Snapshot date",       str(row["snapshot_date"])),
                    ("Role level",          f"{int(row['role_level'])} / 4"),
                    ("Avg rating (all-time)", f"{row['avg_rating_all']:.2f} / 5.0"),
                    ("Absence rate",        f"{row['absence_events_per_month']:.2f} events/month"),
                    ("Absences (last 90d)", f"{int(row['absence_events_last90'])}"),
                ]),
                unsafe_allow_html=True,
            )

        # SHAP chart
        st.markdown('<p class="section-label">Risk Drivers</p>', unsafe_allow_html=True)
        st.plotly_chart(
            shap_waterfall(names, shap_vals, base),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Playbook
        st.markdown('<p class="section-label">Recommended Action</p>', unsafe_allow_html=True)
        st.markdown(playbook_box_html(tier, action, owner), unsafe_allow_html=True)

        # History chart
        if len(emp_rows) > 1:
            st.markdown('<p class="section-label">Risk Score Over Time</p>', unsafe_allow_html=True)
            hist = emp_rows.sort_values("snapshot_date")
            st.plotly_chart(
                history_chart(hist),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ================================================================
    # TAB 2 - At-Risk Leaderboard
    # ================================================================
    with tab2:
        col_filter, col_jump = st.columns([2, 2], gap="large")

        with col_filter:
            dept_filter = st.selectbox(
                "Filter by department",
                ["All"] + sorted(latest_df["department"].unique().tolist()),
                key="leaderboard_dept",
            )

        view  = latest_df if dept_filter == "All" else latest_df[latest_df["department"] == dept_filter]
        top20 = view.sort_values("risk_score", ascending=False).head(20)

        with col_jump:
            jump_options = [
                (int(r["employee_id"]), f"#{int(r['employee_id'])}  {r['department']}  {r['risk_score']:.1%}")
                for _, r in top20.iterrows()
            ]
            jump_id = st.selectbox(
                "Load employee in profile tab",
                options=[x[0] for x in jump_options],
                format_func=lambda v: next(label for eid, label in jump_options if eid == v),
                key="leaderboard_jump",
            )
            if st.button("Load Profile", type="primary"):
                st.session_state["emp_id_input"] = int(jump_id)
                st.toast(f"Employee {int(jump_id)} selected -- switch to Employee Risk Profile tab")
                st.rerun()

        st.markdown(
            f'<p style="font-size:13px;color:#6F767E;margin:16px 0 12px">'
            f'Top {len(top20)} employees by predicted risk &mdash; latest snapshot {latest_date}'
            f'</p>',
            unsafe_allow_html=True,
        )
        st.markdown(leaderboard_html(top20), unsafe_allow_html=True)

        st.markdown(
            '<div class="callout">'
            '<strong>How to read this:</strong> Risk score is the predicted probability of leaving within '
            'six months. Precision at the top 10% is <strong>55.1%</strong>, meaning more than half the '
            'employees flagged in this decile are genuine flight risks.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ================================================================
    # TAB 3 - Department Overview
    # ================================================================
    with tab3:
        dept_opts = ["All"] + sorted(latest_df["department"].unique().tolist())
        dept_sel  = st.selectbox("Filter by department", dept_opts, key="dept_overview_filter")
        dept_view = latest_df if dept_sel == "All" else latest_df[latest_df["department"] == dept_sel]

        c_bar, c_stack = st.columns(2, gap="large")

        with c_bar:
            st.plotly_chart(
                dept_bar_chart(dept_view),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with c_stack:
            st.plotly_chart(
                dept_stack_chart(dept_view),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Summary table
        st.markdown('<p class="section-label">Department Summary</p>', unsafe_allow_html=True)
        dept_stats = (
            dept_view.groupby("department")["risk_score"]
            .agg(
                avg_risk="mean",
                median_risk="median",
                headcount="count",
                high_risk=lambda x: (x > 0.70).sum(),
            )
            .reset_index()
            .sort_values("high_risk", ascending=False)
        )
        dept_stats.columns = ["Department", "Avg Risk", "Median Risk", "Headcount", "High-Risk Count"]
        dept_stats["Avg Risk"]    = dept_stats["Avg Risk"].map("{:.1%}".format)
        dept_stats["Median Risk"] = dept_stats["Median Risk"].map("{:.1%}".format)
        st.dataframe(dept_stats, use_container_width=True, hide_index=True)

    # ----- Fixed footer -----
    st.markdown(
        '<div class="app-footer">'
        'Made by <a href="https://github.com/Mprtham" target="_blank" rel="noopener noreferrer">'
        'Prathamesh Mishra</a>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
