# FinYo Inclusion Engine — 100% IN-MEMORY DEMO (FINAL v8 – FIXED CHARTS)
# Realistic Emirate differences + visible Policy uplift
# =====================================================================

import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json

# -----------------------------------------------------------------------------
# Page config & styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FinYo Inclusion Engine — Synthetic Demo", layout="wide")

# --- Sand/Teal Palette ---
COLOR_BASELINE = "#d4c2a3"   # light sand
COLOR_POLICY = "#6fa79b"    # muted teal
COLOR_HIGHLIGHT = "#e07a5f"  # warm clay
COLOR_MUTED = "#A0B8B8"
BG_COLOR = "#f6f4ef"
TABLE_BORDER = "#E5DCCB"

st.markdown(
    f"""
    <style>
        .stApp {{background-color: {BG_COLOR};}}
        .stDataFrame {{border: 1px solid {TABLE_BORDER};}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# CONFIG & CONSTANTS
# --------------------------------------------------------------------
EMIRATES = ["Abu Dhabi","Dubai","Sharjah","Ajman","Ras Al Khaimah","Fujairah","Umm Al Quwain"]
SECTORS = [
    "Construction","Hospitality / Tourism / F&B","Cleaning & Maintenance","Retail & Supermarkets",
    "Transport & Delivery","Manufacturing & Workshops","Agriculture & Landscaping","Domestic Services",
    "Trade & Repair Services","Business Services / Outsourcing","Real Estate & Property Management",
    "Healthcare & Medical Services","Education & Training Institutions","Financial Services & Insurance",
    "Information Technology & Telecom","Security & Facilities Management",
]

EMPLOYERS_PER_COMBO = 10
WORKERS_PER_EMPLOYER = 8

POLICY_INCOME_BOOST = 0.20
POLICY_SAVINGS_BOOST = 0.10
POLICY_RENT_REDUCTION = 0.05
POLICY_REMIT_REDUCTION = 0.05

# Realistic emirate bias (now amplified so it's visible)
EMIRATE_BIAS = {
    "Abu Dhabi":        {"income": +0.10, "savings": +0.08, "rent": -0.06, "remit": -0.04},
    "Dubai":            {"income": +0.06, "savings": +0.05, "rent": +0.05, "remit": +0.00},
    "Sharjah":          {"income": -0.04, "savings": -0.06, "rent": +0.04, "remit": +0.05},
    "Ajman":            {"income": -0.08, "savings": -0.09, "rent": -0.04, "remit": +0.07},
    "Ras Al Khaimah":   {"income": +0.02, "savings": -0.03, "rent": -0.03, "remit": -0.02},
    "Fujairah":         {"income": -0.05, "savings": -0.04, "rent": +0.02, "remit": +0.04},
    "Umm Al Quwain":    {"income": -0.09, "savings": -0.10, "rent": +0.05, "remit": +0.06},
}

FRI_GRADE_THRESHOLDS = {"A": 0.85, "B": 0.70, "C": 0.50, "D": 0.00}

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def fhi_status(fhi: float) -> str:
    if fhi < 0.45: return "Vulnerable"
    elif fhi < 0.60: return "Watchlist"
    return "Healthy"

def map_fhi_to_band_0_100(fhi: float):
    score = max(0.0, min(100.0, fhi * 100))
    if score < 40: return "Financially Vulnerable", score
    elif score < 70: return "Financially Coping", score
    return "Financially Healthy", score

def grade_from_fri(fri_value: float) -> str:
    if fri_value >= FRI_GRADE_THRESHOLDS["A"]: return "A"
    if fri_value >= FRI_GRADE_THRESHOLDS["B"]: return "B"
    if fri_value >= FRI_GRADE_THRESHOLDS["C"]: return "C"
    return "D"

def build_health_narrative(row: dict) -> str:
    fhi, status = row["FHI"], row["Status"]
    exp_p, liq_b = row["Expense Pressure"], row["Liquidity Buffer"]
    remit, wage = row["Remittance Load"], row["Wage Consistency"]
    band_label, score_100 = map_fhi_to_band_0_100(fhi)

    lines = [
        f"Worker is **{status}** with FHI **{score_100:.0f}/100** ({band_label})."
    ]
    if wage < 0.5: lines.append("Wage income is **volatile** (irregular/delayed payroll).")
    else: lines.append("Wage income is **consistent**.")
    if exp_p > 0.7: lines.append("**High** expense pressure.")
    elif exp_p > 0.5: lines.append("**Moderate** expense pressure.")
    else: lines.append("Expense pressure is **manageable**.")
    if liq_b < 0.3: lines.append("**Low** liquidity buffer.")
    else: lines.append("**Adequate** liquidity buffer.")
    if remit > 0.4: lines.append("**High** remittance outflows.")
    return " ".join(lines)

# --------------------------------------------------------------------
# FIXED: Realistic per-emirate + per-sector randomness
# --------------------------------------------------------------------
def generate_synthetic_features(sector: str, emirate: str) -> dict:
    # Unique but reproducible seed for every emirate × sector combo
    seed = hash(emirate + sector) % (2**32)
    rng = np.random.default_rng(seed)

    # Persona probabilities per sector (unchanged)
    if sector in ["Construction", "Cleaning & Maintenance", "Agriculture & Landscaping", "Domestic Services"]:
        p = {"vulnerable": 0.45, "coping": 0.40, "stable": 0.15, "high_earner": 0.00}
    elif sector in ["Hospitality / Tourism / F&B", "Retail & Supermarkets", "Transport & Delivery"]:
        p = {"vulnerable": 0.25, "coping": 0.50, "stable": 0.20, "high_earner": 0.05}
    elif sector in ["Financial Services & Insurance", "Information Technology & Telecom"]:
        p = {"vulnerable": 0.05, "coping": 0.20, "stable": 0.50, "high_earner": 0.25}
    else:
        p = {"vulnerable": 0.20, "coping": 0.45, "stable": 0.30, "high_earner": 0.05}

    persona = rng.choice(list(p.keys()), p=list(p.values()))

    if persona == "vulnerable":
        base = {"inc": rng.uniform(0.15, 0.45), "sav": rng.uniform(0.00, 0.08),
                "rent": rng.uniform(0.45, 0.65), "rem": rng.uniform(0.35, 0.60),
                "tenure": rng.integers(3, 12), "ded": rng.uniform(0.25, 0.55)}
    elif persona == "coping":
        base = {"inc": rng.uniform(0.40, 0.70), "sav": rng.uniform(0.06, 0.18),
                "rent": rng.uniform(0.30, 0.50), "rem": rng.uniform(0.20, 0.45),
                "tenure": rng.integers(8, 30), "ded": rng.uniform(0.10, 0.35)}
    elif persona == "stable":
        base = {"inc": rng.uniform(0.65, 0.90), "sav": rng.uniform(0.15, 0.35),
                "rent": rng.uniform(0.20, 0.40), "rem": rng.uniform(0.10, 0.30),
                "tenure": rng.integers(18, 48), "ded": rng.uniform(0.05, 0.20)}
    else:  # high_earner
        base = {"inc": rng.uniform(0.80, 0.98), "sav": rng.uniform(0.25, 0.50),
                "rent": rng.uniform(0.15, 0.30), "rem": rng.uniform(0.05, 0.20),
                "tenure": rng.integers(30, 72), "ded": rng.uniform(0.00, 0.10)}

    # Apply strong emirate bias
    bias = EMIRATE_BIAS[emirate]
    features = {
        "income_stability": np.clip(base["inc"] + bias["income"], 0.0, 1.0),
        "savings_ratio":     np.clip(base["sav"] + bias["savings"], 0.0, 1.0),
        "rent_ratio":        np.clip(base["rent"] + bias["rent"], 0.0, 1.0),
        "remittance_ratio":  np.clip(base["rem"] + bias["remit"], 0.0, 1.0),
        "tenure_months":     base["tenure"],
        "deduction_ratio":   base["ded"],
    }
    return features

def derive_metrics(f):
    return {
        "expense_pressure": 1.0 - f["savings_ratio"],
        "liquidity_buffer": f["savings_ratio"],
        "wage_consistency": f["income_stability"],
        "remittance_load":  f["remittance_ratio"],
        "rent_ratio":       f["rent_ratio"],
    }

# --------------------------------------------------------------------
# MAIN SIMULATION (cached)
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def run_scenario(label: str, is_policy: bool):
    worker_rows = []
    emirate_sector_agg = {}

    for emirate in EMIRATES:
        for sector in SECTORS:
            key = (emirate, sector)
            if key not in emirate_sector_agg:
                emirate_sector_agg[key] = {"fhi": [], "total": 0, "vulnerable": 0}

            for emp_idx in range(1, EMPLOYERS_PER_COMBO + 1):
                emp_id = f"{emirate[:3].upper()}_{sector[:3].upper()}_{emp_idx:03d}"

                for w_idx in range(1, WORKERS_PER_EMPLOYER + 1):
                    f = generate_synthetic_features(sector, emirate)

                    # Policy uplift
                    if is_policy:
                        f["income_stability"] = min(1.0, f["income_stability"] + POLICY_INCOME_BOOST)
                        f["savings_ratio"]     = min(1.0, f["savings_ratio"]     + POLICY_SAVINGS_BOOST)
                        f["rent_ratio"]        = max(0.0, f["rent_ratio"]        - POLICY_RENT_REDUCTION)
                        f["remittance_ratio"]  = max(0.0, f["remittance_ratio"]  - POLICY_REMIT_REDUCTION)

                    bh = derive_metrics(f)

                    # FHI with tiny realistic noise
                    fhi = (bh["wage_consistency"] * 0.40 +
                           bh["liquidity_buffer"] * 0.30 +
                           (1.0 - bh["rent_ratio"]) * 0.15 +
                           (1.0 - bh["remittance_load"]) * 0.15 +
                           np.random.normal(0, 0.018))
                    fhi = np.clip(fhi, 0.05, 0.95)

                    # Simple FRI
                    fri = 0.3*(f["tenure_months"]/36.0) + 0.4*bh["wage_consistency"] + 0.3*(1.0 - abs(bh["remittance_load"] - 0.35))
                    fri = np.clip(fri, 0.1, 0.95)

                    row = {
                        "Scenario": label, "Emirate": emirate, "Sector": sector,
                        "Employer ID": emp_id, "FHI": fhi, "FRI": fri,
                        "Status": fhi_status(fhi),
                        "Expense Pressure": round(bh["expense_pressure"], 2),
                        "Liquidity Buffer": round(bh["liquidity_buffer"], 2),
                        "Remittance Load": round(bh["remittance_load"], 2),
                        "Wage Consistency": round(bh["wage_consistency"], 2),
                        "Rent Ratio": round(bh["rent_ratio"], 2),
                    }
                    row["Health Narrative"] = build_health_narrative(row)
                    worker_rows.append(row)

                    # Aggregations
                    emirate_sector_agg[key]["fhi"].append(fhi)
                    emirate_sector_agg[key]["total"] += 1
                    if row["Status"] == "Vulnerable":
                        emirate_sector_agg[key]["vulnerable"] += 1

    # Emirate-Sector summary
    es_rows = []
    for (e, s), agg in emirate_sector_agg.items():
        es_rows.append({
            "Scenario": label, "Emirate": e, "Sector": s,
            "Avg FHI": round(np.mean(agg["fhi"]), 3),
            "% Vulnerable": round(100 * agg["vulnerable"] / agg["total"], 1)
        })
    workers_df = pd.DataFrame(worker_rows)
    es_df = pd.DataFrame(es_rows)
    return workers_df, es_df

# --------------------------------------------------------------------
# RUN SIMULATIONS (only once)
# --------------------------------------------------------------------
if "data" not in st.session_state:
    with st.spinner("Generating realistic synthetic population..."):
        base_workers, base_es = run_scenario("Baseline", False)
        pol_workers,  pol_es  = run_scenario("Policy", True)
        st.session_state.data = {
            "Baseline": {"workers": base_workers, "es": base_es},
            "Policy":   {"workers": pol_workers,  "es": pol_es},
        }

data = st.session_state.data

# --------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------
st.title("FinYo Inclusion Engine – Synthetic Demo")
st.caption("Powered by FinYo Inclusion Engine • 100% synthetic data")

tab1, tab2 = st.tabs(["Financial Health (FHI)", "Alternative Credit (FRI)"])

with tab1:
    st.subheader("National & Emirate Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average FHI by Emirate (Baseline vs Policy)")
        df_em = pd.concat([
            data["Baseline"]["es"].groupby("Emirate")["Avg FHI"].mean().reset_index().assign(Scenario="Baseline"),
            data["Policy"]["es"].groupby("Emirate")["Avg FHI"].mean().reset_index().assign(Scenario="Policy"),
        ])
        fig_bar = px.bar(df_em, y="Emirate", x="Avg FHI", color="Scenario",
                         orientation="h", barmode="group",
                         color_discrete_map={"Baseline": COLOR_BASELINE, "Policy": COLOR_POLICY},
                         text_auto=".3f")
        fig_bar.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### FHI Uplift (Policy − Baseline)")
        merged = data["Baseline"]["es"][["Emirate","Sector","Avg FHI"]].merge(
            data["Policy"]["es"][["Emirate","Sector","Avg FHI"]],
            on=["Emirate","Sector"], suffixes=("_Base", "_Pol"))
        merged["Δ FHI"] = merged["Avg FHI_Pol"] - merged["Avg FHI_Base"]

        fig_bubble = px.scatter(merged, x="Sector", y="Emirate", size="Δ FHI", color="Δ FHI",
                                color_continuous_scale=["#d4c2a3", "#6fa79b"], size_max=45,
                                range_color=[-0.02, 0.15])
        fig_bubble.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Summary metrics
    avg_base = data["Baseline"]["es"]["Avg FHI"].mean()
    avg_pol  = data["Policy"]["es"]["Avg FHI"].mean()
    st.metric("National Avg FHI", f"{avg_base:.3f} → {avg_pol:.3f}", f"+{avg_pol-avg_base:.3f}")

with tab2:
    st.info("FRI / Bank view coming soon in full version. FHI dashboard is fully functional.")

# Footer
st.caption(f"Simulation timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC • "
           "Baseline = today • Policy = simulated wage/savings improvements • Synthetic data only")
