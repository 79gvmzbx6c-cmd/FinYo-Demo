# =====================================================================
# FinYo Inclusion Engine — 100% IN-MEMORY DEMO (FINAL v7 – Emirate Bias)
#
#    • STRATEGY: 100% IN-MEMORY.
#    • NEW: Added EMIRATE_BIAS to the simulation.
#      This creates realistic, different FHI scores for each
#      Emirate, fixing the "flat chart" problem.
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
COLOR_BASELINE = "#d4c2a3"  # light sand
COLOR_POLICY = "#6fa79b"  # muted teal
COLOR_HIGHLIGHT = "#e07a5f"  # warm clay for risk / anomalies
COLOR_MUTED = "#A0B8B8"
BG_COLOR = "#f6f4ef"
TABLE_BORDER = "#E5DCCB"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BG_COLOR};
        }}
        .stDataFrame {{
            border: 1px solid {TABLE_BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# SHARED CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------
EMIRATES = [
    "Abu Dhabi",
    "Dubai",
    "Sharjah",
    "Ajman",
    "Ras Al Khaimah",
    "Fujairah",
    "Umm Al Quwain",
]

SECTORS = [
    "Construction",
    "Hospitality / Tourism / F&B",
    "Cleaning & Maintenance",
    "Retail & Supermarkets",
    "Transport & Delivery",
    "Manufacturing & Workshops",
    "Agriculture & Landscaping",
    "Domestic Services",
    "Trade & Repair Services",
    "Business Services / Outsourcing",
    "Real Estate & Property Management",
    "Healthcare & Medical Services",
    "Education & Training Institutions",
    "Financial Services & Insurance",
    "Information Technology & Telecom",
    "Security & Facilities Management",
]

EMPLOYERS_PER_COMBO = 10
WORKERS_PER_EMPLOYER = 8

POLICY_INCOME_BOOST = 0.20
POLICY_SAVINGS_BOOST = 0.10
POLICY_RENT_REDUCTION = 0.05
POLICY_REMIT_REDUCTION = 0.05

EMIRATE_CODE = {
    "Abu Dhabi": "ABU",
    "Dubai": "DUB",
    "Sharjah": "SHJ",
    "Ajman": "AJM",
    "Ras Al Khaimah": "RAK",
    "Fujairah": "FUJ",
    "Umm Al Quwain": "UAQ",
}

SECTOR_CODE = {
    "Construction": "CON",
    "Hospitality / Tourism / F&B": "HOS",
    "Cleaning & Maintenance": "CLE",
    "Retail & Supermarkets": "RTL",
    "Transport & Delivery": "TRN",
    "Manufacturing & Workshops": "MFG",
    "Agriculture & Landscaping": "AGR",
    "Domestic Services": "DOM",
    "Trade & Repair Services": "TRS",
    "Business Services / Outsourcing": "BIZ",
    "Real Estate & Property Management": "REA",
    "Healthcare & Medical Services": "HEA",
    "Education & Training Institutions": "EDU",
    "Financial Services & Insurance": "FIN",
    "Information Technology & Telecom": "ITT",
    "Security & Facilities Management": "SEC",
}

# --- NEW: Emirate-level behavioural multipliers (realistic differences) ---
EMIRATE_BIAS = {
    "Abu Dhabi":      {"income": +0.08, "savings": +0.04, "rent": -0.03, "remit": -0.02},
    "Dubai":          {"income": +0.04, "savings": +0.03, "rent": +0.04, "remit": +0.00},
    "Sharjah":        {"income": -0.02, "savings": -0.03, "rent": +0.02, "remit": +0.03},
    "Ajman":          {"income": -0.04, "savings": -0.05, "rent": -0.03, "remit": +0.04},
    "Ras Al Khaimah": {"income": +0.01, "savings": -0.02, "rent": -0.02, "remit": -0.03},
    "Fujairah":       {"income": -0.03, "savings": -0.02, "rent": +0.01, "remit": +0.03},
    "Umm Al Quwain":  {"income": -0.05, "savings": -0.05, "rent": +0.02, "remit": +0.04},
}

FRI_GRADE_THRESHOLDS = {
    "A": 0.85,
    "B": 0.70,
    "C": 0.50,
    "D": 0.00,
}

# --------------------------------------------------------------------
# SMALL HELPERS (USED BY FHI + FRI)
# --------------------------------------------------------------------
def safe_get(d, key, default):
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)

def fhi_status(fhi: float) -> str:
    """Map FHI (0–1) to a simple status bucket."""
    if fhi < 0.45:
        return "Vulnerable"
    elif fhi < 0.60:
        return "Watchlist"
    return "Healthy"

def map_fhi_to_band_0_100(fhi: float) -> tuple[str, float]:
    """
    Map FHI 0–1 to 0–100 band with human label, per your RFP:
        0–39   → Financially Vulnerable
        40–69 → Financially Coping
        70–100 → Financially Healthy
    """
    score_100 = max(0.0, min(100.0, fhi * 100))
    if score_100 < 40:
        return "Financially Vulnerable", score_100
    elif score_100 < 70:
        return "Financially Coping", score_100
    return "Financially Healthy", score_100

def grade_from_fri(fri_value: float) -> str:
    """Map FRI (0–1, higher = safer) to A–D band using Annex E logic."""
    fri_value = float(fri_value)
    if fri_value >= FRI_GRADE_THRESHOLDS["A"]:
        return "A"
    if fri_value >= FRI_GRADE_THRESHOLDS["B"]:
        return "B"
    if fri_value >= FRI_GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"

def fri_decision_text(grade: str) -> str:
    """Human lending decision text aligned with Annex E."""
    if grade == "A":
        return "Eligible for maximum loan (≤ 50% of salary)"
    if grade == "B":
        return "Eligible for mid–high loan band"
    if grade == "C":
        return "Eligible with capped loan and remittance-linked repayment"
    # grade D
    return "Eligible with small capped loan; monitor closely"

def build_health_narrative(row: dict) -> str:
    """
    Simple worker financial health narrative (no FHN acronym shown).
    This is a phase-2 style feature but included here for storytelling.
    """
    fhi = row["FHI"]
    status = row["Status"]
    exp_p = row["Expense Pressure"]
    liq_b = row["Liquidity Buffer"]
    remit = row["Remittance Load"]
    wage = row["Wage Consistency"]
    
    band_label, score_100 = map_fhi_to_band_0_100(fhi)
    
    lines = []
    lines.append(
        f"Worker is classified as **{status}** with Financial Health Index of "
        f"**{score_100:.0f}/100** ({band_label})."
    )
    
    if wage < 0.5:
        lines.append(
            "Wage income is **volatile**, indicating irregular or delayed payroll behaviour."
        )
    else:
        lines.append("Wage income appears **consistent**, with limited delays observed.")
    
    if exp_p > 0.7:
        lines.append(
            "Expense pressure is **high**, leaving limited buffer after essential costs."
        )
    elif exp_p > 0.5:
        lines.append("Expense pressure is **moderate**, requiring some budget discipline.")
    else:
        lines.append("Expense pressure is **manageable** relative to income.")
    
    if liq_b < 0.3:
        lines.append(
            "Liquidity buffer is **weak**, suggesting low savings or limited emergency funds."
        )
    else:
        lines.append(
            "Liquidity buffer is **adequate**, indicating some savings capacity."
        )
    
    if remit > 0.4:
        lines.append(
            "High remittance outflows may be constraining local financial resilience."
        )
    
    return " ".join(lines)

def detect_anomalies(worker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple AML-style anomaly detector.
    Flags unusual combinations of FHI, liquidity, remittance etc.
    """
    if worker_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid SettingWithCopyWarning
    worker_df = worker_df.copy()
    
    # Calculate mean FHI for the cohort
    worker_df["FHI_Mean_Cohort"] = worker_df.groupby(["Emirate", "Sector"])["FHI"].transform('mean')
    
    records = []
    for _, r in worker_df.iterrows():
        reasons = []
        base = r["FHI_Mean_Cohort"]
        
        if r["Remittance Load"] > 0.6 and r["Liquidity Buffer"] < 0.2:
            reasons.append("High remittance with low liquidity buffer.")
        
        if r["Expense Pressure"] > 0.8 and r["Wage Consistency"] < 0.4:
            reasons.append("High expense pressure with unstable wages.")
        
        if r["FHI"] < base - 0.10:
            reasons.append("FHI significantly below emirate-sector norm.")
        
        if r["FHI"] < 0.40 and r["Remittance Load"] > 0.5:
            reasons.append("Low FHI with high remittance intensity.")
        
        if reasons:
            records.append(
                {
                    "Scenario": r["Scenario"],
                    "Emirate": r["Emirate"],
                    "Sector": r["Sector"],
                    "Employer ID": r["Employer ID"],
                    "FHI": r["FHI"],
                    "Status": r["Status"],
                    "Suspicion Flag": "Anomaly / Suspicious pattern",
                    "Reasons": " ".join(reasons),
                }
            )
    
    return pd.DataFrame(records)

# --------------------------------------------------------------------
# FHI IN-MEMORY GENERATION (MOHRE VIEW)
# --------------------------------------------------------------------
np.random.seed(42) # Seed for reproducibility

# ====================================================================
# --- THIS IS THE "SMART" SIMULATION FUNCTION ---
# ====================================================================
def generate_synthetic_features_internal(sector: str) -> dict:
    """
    Generates realistic feature clusters based on 4 worker personas.
    Probabilities are now based on the SECTOR.
    """
    
    # Define persona probabilities
    if sector in ["Construction", "Cleaning & Maintenance", "Agriculture & Landscaping", "Domestic Services"]:
        # Higher vulnerability sectors
        personas = {
            "vulnerable": 0.45,
            "coping": 0.45,
            "stable": 0.10,
            "high_earner": 0.00,
        }
    elif sector in ["Hospitality / Tourism / F&B", "Retail & Supermarkets", "Transport & Delivery", "Manufacturing & Workshops"]:
        # Middle / Coping sectors
        personas = {
            "vulnerable": 0.20,
            "coping": 0.50,
            "stable": 0.25,
            "high_earner": 0.05,
        }
    elif sector in ["Financial Services & Insurance", "Information Technology & Telecom", "Healthcare & Medical Services", "Education & Training Institutions"]:
        # Higher stability sectors
        personas = {
            "vulnerable": 0.05,
            "coping": 0.25,
            "stable": 0.50,
            "high_earner": 0.20,
        }
    else:
        # Default for other sectors
        personas = {
            "vulnerable": 0.20,
            "coping": 0.40,
            "stable": 0.30,
            "high_earner": 0.10,
        }
    
    persona = np.random.choice(list(personas.keys()), p=list(personas.values()))
    
    if persona == "vulnerable":
        features = {
            "income_stability": np.random.uniform(0.1, 0.4),
            "savings_ratio":    np.random.uniform(0.0, 0.05),
            "rent_ratio":       np.random.uniform(0.4, 0.6),
            "remittance_ratio": np.random.uniform(0.3, 0.5),
            "tenure_months":    np.random.randint(3, 12),
            "deduction_ratio":  np.random.uniform(0.2, 0.5),
        }
    elif persona == "coping":
        features = {
            "income_stability": np.random.uniform(0.4, 0.7),
            "savings_ratio":    np.random.uniform(0.05, 0.15),
            "rent_ratio":       np.random.uniform(0.3, 0.45),
            "remittance_ratio": np.random.uniform(0.2, 0.4),
            "tenure_months":    np.random.randint(6, 24),
            "deduction_ratio":  np.random.uniform(0.1, 0.3),
        }
    elif persona == "stable":
        features = {
            "income_stability": np.random.uniform(0.7, 0.9),
            "savings_ratio":    np.random.uniform(0.15, 0.3),
            "rent_ratio":       np.random.uniform(0.25, 0.35),
            "remittance_ratio": np.random.uniform(0.1, 0.3),
            "tenure_months":    np.random.randint(12, 36),
            "deduction_ratio":  np.random.uniform(0.05, 0.2),
        }
    else:  # high_earner
        features = {
            "income_stability": np.random.uniform(0.85, 1.0),
            "savings_ratio":    np.random.uniform(0.25, 0.5),
            "rent_ratio":       np.random.uniform(0.2, 0.3),
            "remittance_ratio": np.random.uniform(0.05, 0.2),
            "tenure_months":    np.random.randint(24, 60),
            "deduction_ratio":  np.random.uniform(0.0, 0.1),
        }
    
    return features

# ====================================================================
# --- END OF "SMART" FUNCTION ---
# ====================================================================

def derive_behavioural_metrics(features: dict) -> dict:
    """Derive high-level metrics from raw feature vector."""
    income_stability = safe_get(features, "income_stability", 0.7)
    savings_ratio = safe_get(features, "savings_ratio", 0.25)
    rent_ratio = safe_get(features, "rent_ratio", 0.30)
    remittance_ratio = safe_get(features, "remittance_ratio", 0.25)
    
    expense_pressure = min(1.0, 1.0 - savings_ratio)
    liquidity_buffer = max(0.0, savings_ratio)
    wage_consistency = max(0.0, min(1.0, income_stability))
    remittance_load = max(0.0, min(1.0, remittance_ratio))
    
    return {
        "expense_pressure": expense_pressure,
        "liquidity_buffer": liquidity_buffer,
        "wage_consistency": wage_consistency,
        "remittance_load": remittance_load,
        "rent_ratio": rent_ratio,
    }

# --- This is the new, fast, in-memory simulation function ---
# STEP 3: Changed cache decorator
@st.cache_data(show_spinner=False, ttl=1)
def run_full_scenario_internal(label: str, is_policy: bool):
    """
    Generate full UAE simulation for one scenario (Baseline OR Policy).
    Returns:
        workers_df, employers_df, emirate_sector_df, summary_dict, anomalies_df
    """
    worker_rows = []
    employer_rows = []
    emirate_sector_agg = {}
    
    for emirate in EMIRATES:
        for sector in SECTORS:
            combo_key = (emirate, sector)
            
            if combo_key not in emirate_sector_agg:
                emirate_sector_agg[combo_key] = {
                    "fhi_scores": [],
                    "fri_scores": [],
                    "vulnerable": 0,
                    "total": 0,
                }
            
            for emp_idx in range(1, EMPLOYERS_PER_COMBO + 1):
                emp_id = f"{EMIRATE_CODE[emirate]}_{SECTOR_CODE.get(sector, 'UNK')}_{emp_idx:03d}"
                
                emp_fhis = []
                emp_fris = []
                emp_payroll_delays = []
                emp_vulnerable = 0
                
                for worker_idx in range(1, WORKERS_PER_EMPLOYER + 1):
                    
                    # 1. Generate realistic persona-based features
                    # --- THIS IS THE FIRST FIX ---
                    features = generate_synthetic_features_internal(sector=sector)
                    
                    # --- THIS IS THE SECOND FIX ---
                    # --- Apply Emirate-level bias to features (makes Emirates DIFFERENT) ---
                    bias = EMIRATE_BIAS[emirate]
                    features["income_stability"] = np.clip(features["income_stability"] + bias["income"], 0, 1)
                    features["savings_ratio"]    = np.clip(features["savings_ratio"] + bias["savings"], 0, 1)
                    features["rent_ratio"]       = np.clip(features["rent_ratio"] + bias["rent"], 0, 1)
                    features["remittance_ratio"] = np.clip(features["remittance_ratio"] + bias["remit"], 0, 1)
                    
                    # 2. Apply Policy uplift if needed
                    if is_policy:
                        inc = safe_get(features, "income_stability", 0.7)
                        sav = safe_get(features, "savings_ratio", 0.25)
                        rent = safe_get(features, "rent_ratio", 0.30)
                        rem = safe_get(features, "remittance_ratio", 0.25)
                        # --- FIX: Use multiplicative boosts, not additive ---
                        # This makes the uplift proportional to the starting value
                        features["income_stability"] = min(1.0, inc * (1 + POLICY_INCOME_BOOST))
                        features["savings_ratio"] = min(1.0, sav * (1 + POLICY_SAVINGS_BOOST))
                        features["rent_ratio"] = max(0.0, rent * (1 - POLICY_RENT_REDUCTION))
                        features["remittance_ratio"] = max(0.0, rem * (1 - POLICY_REMIT_REDUCTION))
                    
                    # 3. Derive metrics
                    bh = derive_behavioural_metrics(features)
                    
                    # 4. Generate scores (simple model for demo)
                    fhi = (
                        (bh["wage_consistency"] * 0.4) +
                        (bh["liquidity_buffer"] * 0.3) +
                        ((1.0 - bh["rent_ratio"]) * 0.15) +
                        ((1.0 - bh["remittance_load"]) * 0.15)
                    )
                    fhi = np.clip(fhi, 0.05, 0.95)
                    
                    # --- REALISTIC FRI ---
                    contract_score = min(1.0, features["tenure_months"] / 24.0)
                    employer_score = bh["wage_consistency"]
                    remittance_sub = (1.0 - abs(bh["remittance_load"] - 0.35) / 0.35)  # best near 35%
                    obligation_sub = (1.0 - features["deduction_ratio"])
                    
                    fri = (
                        (contract_score * 0.3) +
                        (employer_score * 0.3) +
                        (remittance_sub * 0.2) +
                        (obligation_sub * 0.2)
                    )
                    fri = np.clip(fri, 0.05, 0.95)  # risk score, 0.85+ is good
                    
                    frx = fhi * 0.5 + fri * 0.5
                    status = fhi_status(fhi)
                    exp_p = bh["expense_pressure"]
                    liq_b = bh["liquidity_buffer"]
                    wage = bh["wage_consistency"]
                    remit = bh["remittance_load"]
                    rent = bh["rent_ratio"]
                    
                    signals = []
                    if wage < 0.5:
                        signals.append("Unstable wages / WPS delays")
                    if exp_p > 0.7:
                        signals.append("High expense pressure")
                    if liq_b < 0.3:
                        signals.append("Low liquidity buffer")
                    if remit > 0.4:
                        signals.append("High remittance outflow")
                    if rent > 0.4:
                        signals.append("High rent burden")
                    
                    # Simplified Reason Codes
                    flat_reasons = []
                    if fhi < 0.4: flat_reasons.append("LOW_FHI_SCORE")
                    if wage < 0.5: flat_reasons.append("INCOME_STABILITY_WEAK")
                    if liq_b < 0.2: flat_reasons.append("LOW_LIQUIDITY_BUFFER")
                    if features["tenure_months"] < 6: flat_reasons.append("TENURE_SHORT")
                    if features["deduction_ratio"] > 0.35: flat_reasons.append("HIGH_DEDUCTION_PRESSURE")
                    if not flat_reasons: flat_reasons.append("STABLE_PROFILE")
                    
                    delay_days = max(0.0, (1.0 - wage) * 20.0)
                    
                    row = {
                        "Scenario": label,
                        "Emirate": emirate,
                        "Sector": sector,
                        "Employer ID": emp_id,
                        "Worker Index": (emp_idx - 1) * WORKERS_PER_EMPLOYER + worker_idx,
                        "FHI": fhi,
                        "FRI": fri,
                        "FRX": frx,
                        "Status": status,
                        "Expense Pressure": round(exp_p, 2),
                        "Liquidity Buffer": round(liq_b, 2),
                        "Remittance Load": round(remit, 2),
                        "Wage Consistency": round(wage, 2),
                        "Rent Ratio": round(rent, 2),
                        "Behavioural Signals": "; ".join(signals) if signals else "—",
                        "AI Reason Codes": "; ".join(sorted(set(flat_reasons))) if flat_reasons else "—",
                        "Payroll Delay (days)": round(delay_days, 1),
                        # Add raw features for FRI tab
                        "Monthly Salary (AED)": np.random.choice(
                            [1500, 2000, 2300, 3000, 4000, 5000],
                            p=[0.25, 0.20, 0.25, 0.10, 0.10, 0.10],
                        ),
                        "Tenure (months)": features["tenure_months"],
                        "Deduction Ratio": features["deduction_ratio"],
                    }
                    
                    row["Health Narrative"] = build_health_narrative(row)
                    
                    worker_rows.append(row)
                    emp_fhis.append(fhi)
                    emp_fris.append(fri)
                    emp_payroll_delays.append(delay_days)
                    
                    if status == "Vulnerable":
                        emp_vulnerable += 1
                    
                    emirate_sector_agg[combo_key]["fhi_scores"].append(fhi)
                    emirate_sector_agg[combo_key]["fri_scores"].append(fri)
                    emirate_sector_agg[combo_key]["total"] += 1
                    if status == "Vulnerable":
                        emirate_sector_agg[combo_key]["vulnerable"] += 1
                
                # Employer-level aggregation
                if not emp_fhis:
                    continue
                
                emp_avg_fhi = float(np.mean(emp_fhis))
                emp_avg_fri = float(np.mean(emp_fris))
                emp_avg_delay = float(np.mean(emp_payroll_delays))
                emp_vul_pct = 100.0 * emp_vulnerable / WORKERS_PER_EMPLOYER
                
                risk_bucket = (
                    "High Risk"
                    if emp_avg_fhi < 0.45
                    else "Medium Risk"
                    if emp_avg_fhi < 0.60
                    else "Low Risk"
                )
                
                oversight_rec = (
                    "Flag for inspection / audit"
                    if risk_bucket == "High Risk"
                    else "Send reminders / incentives"
                    if risk_bucket == "Medium Risk"
                    else "Model employer – monitoring only"
                )
                
                employer_rows.append(
                    {
                        "Scenario": label,
                        "Employer ID": emp_id,
                        "Emirate": emirate,
                        "Sector": sector,
                        "Avg FHI": round(emp_avg_fhi, 2),
                        "Avg FRI": round(emp_avg_fri, 2),
                        "% Vulnerable Workers": round(emp_vul_pct, 1),
                        "Avg Payroll Delay (days)": round(emp_avg_delay, 1),
                        "Oversight Category": risk_bucket,
                        "Oversight Recommendation": oversight_rec,
                    }
                )
    
    # Emirate × Sector overview
    emirate_sector_rows = []
    for (emirate, sector), agg in emirate_sector_agg.items():
        if agg["total"] == 0:
            continue
        
        avg_fhi = float(np.mean(agg["fhi_scores"]))
        avg_fri = float(np.mean(agg["fri_scores"]))
        vul_pct = 100.0 * agg["vulnerable"] / agg["total"]
        
        emirate_sector_rows.append(
            {
                "Scenario": label,
                "Emirate": emirate,
                "Sector": sector,
                "Avg FHI": round(avg_fhi, 2),
                "Avg FRI": round(avg_fri, 2),
                "% Vulnerable": round(vul_pct, 1),
                "Total Workers": agg["total"],
            }
        )
    
    workers_df = pd.DataFrame(worker_rows)
    employers_df = pd.DataFrame(employer_rows)
    emirate_sector_df = pd.DataFrame(emirate_sector_rows)
    
    # Simple national summary
    summary = {}
    if not workers_df.empty:
        summary["avg_fhi"] = workers_df["FHI"].mean()
        summary["total_workers"] = len(workers_df)
        summary["vulnerable_pct"] = (
            (workers_df["Status"] == "Vulnerable").mean() * 100.0
        )
    else:
        summary["avg_fhi"] = 0.0
        summary["total_workers"] = 0
        summary["vulnerable_pct"] = 0.0
    
    anomalies_df = detect_anomalies(workers_df)
    
    return workers_df, employers_df, emirate_sector_df, summary, anomalies_df

# --- Report Generation Helpers for Download Button ---
def make_individual_report_row(row: pd.Series) -> dict:
    """Builds a clean JSON-ready report dict (no PII)."""
    fhi_val = row.get("FHI")
    fri_val = row.get("FRI")
    fhi_band, fhi_score_100 = map_fhi_to_band_0_100(fhi_val)
    
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), # Use timezone-aware datetime
        "scenario": row.get("Scenario",""),
        "emirate": row.get("Emirate",""),
        "sector": row.get("Sector",""),
        "employer_id": row.get("Employer ID",""),
        "worker_index": int(row.get("Worker Index", 0)),
        "scores": {
            "FHI": float(fhi_val),
            "FRI": float(fri_val),
            "FRX": float(row.get("FRX",0.0)),
        },
        "bands": {
            "fhi_band": fhi_band,
            "fri_grade": grade_from_fri(float(fri_val)), # Use the correct grading function
        },
        "behaviour_metrics": {
            "expense_pressure": float(row.get("Expense Pressure",0.0)),
            "liquidity_buffer": float(row.get("Liquidity Buffer",0.0)),
            "remittance_load": float(row.get("Remittance Load",0.0)),
            "wage_consistency": float(row.get("Wage Consistency",0.0)),
            "rent_ratio": float(row.get("Rent Ratio",0.0)),
        },
        "reason_codes": str(row.get("AI Reason Codes","—")),
        "est_payroll_delay_days": float(row.get("Payroll Delay (days)",0.0)),
        "lawful_purpose": "financial_behavior_simulation (synthetic)",
        "disclaimer": "Synthetic demonstration only. No personal data processed.",
    }

def printable_text_report(report_dict: dict) -> str:
    s = report_dict
    lines = []
    lines.append("FinYo Inclusion Engine – Individual Report (Synthetic)")
    lines.append(f"Generated: {s['generated_at']}")
    lines.append(f"Scenario: {s['scenario']}")
    lines.append(f"Emirate / Sector: {s['emirate']} / {s['sector']}")
    lines.append(f"Employer: {s['employer_id']}    Worker Index: {s['worker_index']}")
    lines.append("--------------------------------------------------")
    lines.append(f"FHI: {s['scores']['FHI']:.2f}  ({s['bands']['fhi_band']})")
    lines.append(f"FRI: {s['scores']['FRI']:.2f}  (Grade {s['bands']['fri_grade']})")
    lines.append(f"FRX: {s['scores']['FRX']:.2f}")
    bm = s['behaviour_metrics']
    lines.append("--------------------------------------------------")
    lines.append("Behavioural Metrics")
    lines.append(f"  • Expense Pressure:  {bm['expense_pressure']:.2f}")
    lines.append(f"  • Liquidity Buffer:  {bm['liquidity_buffer']:.2f}")
    lines.append(f"  • Remittance Load:   {bm['remittance_load']:.2f}")
    lines.append(f"  • Wage Consistency:  {bm['wage_consistency']:.2f}")
    lines.append(f"  • Rent Ratio:        {bm['rent_ratio']:.2f}")
    lines.append("--------------------------------------------------")
    lines.append(f"Reason Codes: {s['reason_codes']}")
    lines.append(f"Estimated Payroll Delay (days): {s['est_payroll_delay_days']:.1f}")
    lines.append("--------------------------------------------------")
    lines.append("Lawful Purpose: " + s["lawful_purpose"])
    lines.append("Disclaimer: " + s["disclaimer"])
    return "\n".join(lines)

# --------------------------------------------------------------------
# FRI PORTFOLIO GENERATION (BANK VIEW)
# --------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1)
def build_fri_portfolio_from_cohort(workers_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Takes the FHI worker cohort and enriches it with
    synthetic bank-specific data (salary, loan cap) to
    create the FRI portfolio.
    """
    np.random.seed(42)
    
    if workers_df.empty:
        return pd.DataFrame(), {}
    
    # We only care about the "Policy" scenario for the bank view
    df = workers_df[workers_df["Scenario"] == "Policy"].copy()
    
    borrower_rows = []
    
    # Simple pipeline status
    pipeline_status = np.random.choice(
        [
            "New request",
            "Pending underwriting",
            "Approved – loan offer",
            "Issued – active loan",
        ],
        p=[0.15, 0.20, 0.25, 0.40],
        size=len(df)
    )
    df["Pipeline Status"] = pipeline_status
    
    # Generate FRI-specific fields
    for _, row in df.iterrows():
        fri_01 = row["FRI"]
        grade = grade_from_fri(fri_01)
        salary = row["Monthly Salary (AED)"]
        payroll_delay = row["Payroll Delay (days)"]
        
        # --- Assign Repayment Behaviour based on Risk Grade ---
        if grade == "A":
            behaviour = np.random.choice(
                ["On-Time", "Early", "Partial"], p=[0.85, 0.10, 0.05]
            )
        elif grade == "B":
            behaviour = np.random.choice(
                ["On-Time", "Early", "Partial", "Deferred"], p=[0.70, 0.10, 0.15, 0.05]
            )
        elif grade == "C":
            behaviour = np.random.choice(
                ["On-Time", "Partial", "Deferred"], p=[0.50, 0.35, 0.15]
            )
        else:  # Grade D
            behaviour = np.random.choice(
                ["Partial", "Deferred", "On-Time"], p=[0.50, 0.40, 0.10]
            )
        
        risk_map = {
            "On-Time": "Low",
            "Early": "Low",
            "Partial": "Medium",
            "Deferred": "High",
        }
        risk = risk_map[behaviour]
        
        # --- START OF PROACTIVE ALERT LOGIC ---
        proactive_alert = "—" # Default is no alert
        if payroll_delay > 7:
            proactive_alert = f"PAYROLL DELAY ({payroll_delay:.0f} days)"
            # If pay is late, the *actual* risk is high, even if they paid on time last month
            risk = "High (Proactive)"
        
        elif payroll_delay > 3:
            proactive_alert = f"PAYROLL DELAY ({payroll_delay:.0f} days)"
            risk = "Medium (Proactive)"
        # --- END OF PROACTIVE ALERT LOGIC ---
        
        # Loan band (synthetic) and maximum allowed cap (50% salary)
        raw_cap = 0.5 * salary
        if grade == "B":
            raw_cap *= 0.9
        elif grade == "C":
            raw_cap *= 0.7
        elif grade == "D":
            raw_cap *= 0.5
        
        loan_cap = round(raw_cap, 0)
        
        # Assign synthetic loan band for illustration
        if loan_cap <= 1000:
            loan_band = "300–1 000"
        elif loan_cap <= 3000:
            loan_band = "1 001–3 000"
        else:
            loan_band = "3 001–4 500"
        
        decision_text = fri_decision_text(grade)
        
        # Copy over the live data
        new_row = row.to_dict()
        new_row["Borrower ID"] = f"{row['Employer ID']}-{row['Worker Index']}"
        new_row["FRI Score (0–100)"] = round(fri_01 * 100, 1)
        new_row["Risk Grade (A–D)"] = grade
        new_row["Loan Band (AED)"] = loan_band
        new_row["Loan Cap (AED)"] = loan_cap
        new_row["Decision"] = decision_text
        new_row["Repayment Behaviour"] = behaviour
        new_row["Repayment Risk"] = risk
        new_row["Proactive Alert"] = proactive_alert
        
        borrower_rows.append(new_row)
    
    fri_df = pd.DataFrame(borrower_rows)
    
    summary = {}
    if not fri_df.empty:
        summary["avg_fri"] = fri_df["FRI"].mean()
        summary["avg_score_100"] = fri_df["FRI Score (0–100)"].mean()
        summary["count_low_mod"] = int(
            (fri_df["Risk Grade (A–D)"].isin(["A", "B"])).sum()
        )
        summary["count_high"] = int((fri_df["Risk Grade (A–D)"].isin(["C", "D"])).sum())
        summary["total"] = len(fri_df)
    else:
        summary = {"avg_fri": 0.0, "avg_score_100": 0.0, "count_low_mod": 0, "count_high": 0, "total": 0}
    
    return fri_df, summary

# --------------------------------------------------------------------
# STREAMLIT APP – TOP LEVEL
# --------------------------------------------------------------------
st.title("FinYo Inclusion Engine – Synthetic FHI & FRI Demo")
st.caption(
    "Powered by FinYo Inclusion Engine"
)

# -------------------------------------------------------------
# STEP 2: AUTO-RUN SIMULATION (Baseline + Policy)
# This ensures the demo works instantly on Streamlit Cloud.
# -------------------------------------------------------------
if "fhi_results" not in st.session_state:
    # Baseline scenario
    (
        w_base,
        e_base,
        es_base,
        s_base,
        a_base,
    ) = run_full_scenario_internal("Baseline", is_policy=False)
    
    # Policy scenario
    (
        w_pol,
        e_pol,
        es_pol,
        s_pol,
        a_pol,
    ) = run_full_scenario_internal("Policy", is_policy=True)
    
    # Save into session_state
    st.session_state["fhi_results"] = {
        "Baseline": {
            "workers": w_base,
            "employers": e_base,
            "emirate_sector": es_base,
            "summary": s_base,
            "anomalies": a_base,
        },
        "Policy": {
            "workers": w_pol,
            "employers": e_pol,
            "emirate_sector": es_pol,
            "summary": s_pol,
            "anomalies": a_pol,
        },
    }
    
    # Build and cache FRI data at the same time
    all_workers_df = pd.concat([w_base, w_pol], ignore_index=True)
    fri_port, fri_summary = build_fri_portfolio_from_cohort(all_workers_df)
    st.session_state["fri_results"] = {
        "portfolio": fri_port,
        "summary": fri_summary,
    }

# Unpack for easier use
baseline = st.session_state["fhi_results"]["Baseline"]
policy = st.session_state["fhi_results"]["Policy"]

tab_fhi, tab_fri = st.tabs(
    ["Financial Health (FHI)", "Alternative Credit – Bank View (FRI)"]
)

# ====================================================================
# TAB 1 – FHI / MOHRE PROACTIVE WELFARE VIEW
# ====================================================================
with tab_fhi:
    st.subheader("Financial Health Dashboard")
    st.caption(
        "This view demonstrate **Baseline** (today) vs **Policy** (improved wage stability and support) "
        "across all 7 Emirates and key sectors. The same synthetic workers are scored "
        "under both scenarios to show the impact on Financial Health."
    )
    
    with st.expander("How to read this Baseline vs Policy view", expanded=False):
        st.markdown(
            """
            **Baseline** represents today's behaviour – current wage stability, savings and remittance ratios.
            **Policy** is a *simulated* future where:
            - wage stability improves (better WPS enforcement),
            - savings increase slightly,
            - housing / cost pressure is reduced,
            - remittance outflows are slightly lower.
            
            The **same synthetic workers** are scored under both scenarios.
            
            The difference between them is the impact of MOHRE policy on:
            - Financial Health Index (FHI),
            - Financial Risk & Resilience (FRX),
            - % Vulnerable workers by Emirate, sector, employer.
            """
        )
    
    st.divider()
    
    results = st.session_state["fhi_results"]
    base_sum = results["Baseline"]["summary"]
    pol_sum = results["Policy"]["summary"]
    
    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Avg FHI (Baseline)", f"{base_sum['avg_fhi']:.2f}")
        with k2:
            st.metric(
                "Avg FHI (Policy)",
                f"{pol_sum['avg_fhi']:.2f}",
                delta=f"{pol_sum['avg_fhi'] - base_sum['avg_fhi']:.2f}",
            )
        with k3:
            st.metric(
                "% Vulnerable Workers (Baseline → Policy)",
                f"{base_sum['vulnerable_pct']:.1f}% → {pol_sum['vulnerable_pct']:.1f}%",
                delta=f"{pol_sum['vulnerable_pct'] - base_sum['vulnerable_pct']:.1f} pp",
                delta_color="inverse"
            )
        with k4:
            st.metric("Total Workers Simulated", f"{int(pol_sum['total_workers']):,}")
    
    # Inner tabs for FHI view
    tab_nat, tab_emp, tab_worker, tab_sec = st.tabs(
        [
            "National & Emirate Overview",
            "Employer Oversight",
            "Worker View & Narratives",
            "AML & Risk Monitoring",
        ]
    )
    
    # ----------------------- NATIONAL TAB ----------------------- #
    with tab_nat:
        # Precompute emirate/sector aggregates
        es_base = results["Baseline"]["emirate_sector"]
        es_pol = results["Policy"]["emirate_sector"]
        
        col_chart_1, col_chart_2 = st.columns(2)
        
        # --- Horizontal bar for Emirate averages (Baseline vs Policy) ---
        with col_chart_1:
            with st.container(border=True):
                st.subheader("Average FHI by Emirate (Baseline vs Policy)")
                base_em = es_base.groupby("Emirate")["Avg FHI"].mean().reset_index()
                pol_em = es_pol.groupby("Emirate")["Avg FHI"].mean().reset_index()
                base_em["Scenario"] = "Baseline"
                pol_em["Scenario"] = "Policy"
                em_all = pd.concat([base_em, pol_em])
                
                fig_em = px.bar(
                    em_all,
                    y="Emirate",
                    x="Avg FHI",
                    color="Scenario",
                    orientation="h",
                    barmode="group",
                    color_discrete_map={"Baseline": COLOR_BASELINE, "Policy": COLOR_POLICY},
                    text="Avg FHI",
                )
                fig_em.update_traces(texttemplate='%{text:.2f}', textposition='inside')
                fig_em.update_layout(
                    yaxis_title="Emirate",
                    xaxis_title="Average FHI (0–1)",
                    template="simple_white",
                    legend_title="Scenario",
                )
                fig_em.update_yaxes(categoryorder="category ascending")
                # 🔥 FIX: force real FHI scale
                fig_em.update_xaxes(range=[0, 1])
                st.plotly_chart(fig_em, use_container_width=True)
        
        # --- Bubble Grid for Emirate × Sector FHI uplift (Policy – Baseline) ---
        with col_chart_2:
            with st.container(border=True):
                st.subheader("FHI Uplift (Policy – Baseline)")
                # Merge baseline & policy
                merged = es_base.merge(
                    es_pol,
                    on=["Emirate", "Sector"],
                    suffixes=("_Baseline", "_Policy"),
                )
                merged["FHI Uplift"] = merged["Avg FHI_Policy"] - merged["Avg FHI_Baseline"]
                
                # Filter out any invalid or negative uplift values for bubble sizing
                merged_clean = merged[merged["FHI Uplift"] > 0].copy()
                
                if merged_clean.empty:
                    st.info("No positive FHI uplift data to display.")
                else:
                    # Bubble grid scatter plot
                    fig_bubble = px.scatter(
                        merged_clean,
                        x="Sector",
                        y="Emirate",
                        size="FHI Uplift",
                        color="FHI Uplift",
                        color_continuous_scale=["#f1e3d3", "#d4c2a3", "#6fa79b"],
                        size_max=40,
                    )
                    fig_bubble.update_layout(
                        template="simple_white",
                        xaxis_title="Sector",
                        yaxis_title="Emirate",
                        height=500,
                        coloraxis_colorbar_title="Δ FHI",
                    )
                    fig_bubble.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # ----------------------- EMPLOYER TAB ----------------------- #
    with tab_emp:
        with st.container(border=True):
            st.subheader("Employer Reliability & Policy Oversight")
            scenario_choice = st.radio(
                "Select scenario to view",
                ["Baseline", "Policy"],
                horizontal=True,
                key="emp_scenario",
            )
            
            emp_df = results[scenario_choice]["employers"].copy()
            
            # Filters in expander
            with st.expander("Show Filters"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    emirate_filter = st.multiselect(
                        "Filter by Emirate", EMIRATES, default=None
                    )
                with col_b:
                    sector_filter = st.multiselect(
                        "Filter by Sector", SECTORS, default=None
                    )
                with col_c:
                    risk_filter = st.multiselect(
                        "Filter by Oversight Category",
                        ["Low Risk", "Medium Risk", "High Risk"],
                        default=None,
                    )
            
            if emirate_filter:
                emp_df = emp_df[emp_df["Emirate"].isin(emirate_filter)]
            if sector_filter:
                emp_df = emp_df[emp_df["Sector"].isin(sector_filter)]
            if risk_filter:
                emp_df = emp_df[emp_df["Oversight Category"].isin(risk_filter)]
            
            st.markdown("#### Employer Table")
            st.dataframe(
                emp_df.drop(columns=["Avg FRI"], errors='ignore').sort_values(["Emirate", "Sector", "Employer ID"]),
                use_container_width=True,
                height=350,
            )
        
        # Charts under Employer tab
        col_chart_3, col_chart_4 = st.columns(2)
        
        # Employer FHI leaderboard
        with col_chart_3:
            with st.container(border=True):
                st.markdown("#### Employer FHI Leaderboard (Top 30)")
                if emp_df.empty:
                    st.info("No employers.")
                else:
                    ranked = (
                        emp_df.sort_values("Avg FHI", ascending=False)
                        .head(30)
                        .reset_index(drop=True)
                    )
                    ranked.index = ranked.index + 1
                    ranked["Rank"] = ranked.index
                    
                    leaderboard_cols = [
                        "Rank",
                        "Employer ID",
                        "Emirate",
                        "Sector",
                        "Avg FHI",
                        "Oversight Category",
                    ]
                    st.dataframe(
                        ranked[leaderboard_cols],
                        use_container_width=True,
                        height=520,
                    )
        
        # Employers by Oversight Category
        with col_chart_4:
            with st.container(border=True):
                st.markdown("#### Employers by Oversight Category")
                if emp_df.empty:
                    st.info("No employers to plot.")
                else:
                    fig_emp = px.histogram(
                        emp_df,
                        x="Oversight Category",
                        color="Oversight Category",
                        color_discrete_map={
                            "Low Risk": COLOR_POLICY,
                            "Medium Risk": COLOR_BASELINE,
                            "High Risk": COLOR_HIGHLIGHT,
                        },
                    )
                    fig_emp.update_layout(
                        template="simple_white",
                        xaxis_title="Oversight Category",
                        yaxis_title="Number of employers",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_emp, use_container_width=True)
    
    # ----------------------- WORKER TAB ------------------------- #
    with tab_worker:
        with st.container(border=True):
            st.subheader("Worker View & Explainability")
            scenario_choice_w = st.radio(
                "Scenario", ["Baseline", "Policy"], horizontal=True, key="worker_scenario"
            )
            
            w_df = results[scenario_choice_w]["workers"].copy()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                emirate_choice = st.selectbox("Emirate", ["All"] + EMIRATES)
            with col2:
                sector_choice = st.selectbox("Sector", ["All"] + SECTORS)
            with col3:
                employer_choice = st.selectbox(
                    "Employer ID", ["All"] + sorted(w_df["Employer ID"].unique())
                )
            
            if emirate_choice != "All":
                w_df = w_df[w_df["Emirate"] == emirate_choice]
            if sector_choice != "All":
                w_df = w_df[w_df["Sector"] == sector_choice]
            if employer_choice != "All":
                w_df = w_df[w_df["Employer ID"] == employer_choice]
            
            cL, cR = st.columns([0.6, 0.4])
            
            with cL:
                st.markdown("#### Worker Sample Table")
                cols_to_show = [
                    "Scenario", "Emirate", "Sector", "Employer ID", "Worker Index", "FHI",
                    "FRX", "Status", "Expense Pressure", "Liquidity Buffer",
                    "Remittance Load", "Wage Consistency", "Behavioural Signals", "AI Reason Codes",
                ]
                st.dataframe(
                    w_df[cols_to_show].head(200),
                    use_container_width=True,
                    height=500,
                )
            
            with cR:
                st.markdown("#### Detailed Financial Health Narrative")
                if not w_df.empty:
                    selected_worker = st.selectbox(
                        "Select worker index for narrative",
                        sorted(w_df["Worker Index"].unique()),
                    )
                    
                    wr = (
                        w_df[w_df["Worker Index"] == selected_worker]
                        .iloc[0]
                        .to_dict()
                    )
                    
                    band_label, score_100 = map_fhi_to_band_0_100(wr["FHI"])
                    
                    report_dict = make_individual_report_row(
                        w_df[w_df["Worker Index"] == selected_worker].iloc[0]
                    )
                    
                    st.download_button(
                        "Download JSON Report",
                        data=json.dumps(report_dict, indent=2),
                        file_name=f"report_worker_{selected_worker}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    
                    txt = printable_text_report(report_dict).encode("utf-8")
                    st.download_button(
                        "Download Printable Text",
                        data=txt,
                        file_name=f"report_worker_{selected_worker}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                    
                    st.divider()
                    st.markdown(
                        f"""
                        **Emirate:** {wr['Emirate']} · **Sector:** {wr['Sector']} · **Employer:** {wr['Employer ID']}
                        
                        **FHI:** {wr['FHI']:.2f} ({score_100:.0f}/100, {band_label}) · **FRX:** {wr['FRX']:.2f} · **Status:** {wr['Status']}
                        """
                    )
                    st.markdown("**Behavioural Signals:** " + wr["Behavioural Signals"])
                    st.markdown("**Model Reason Codes:** " + wr["AI Reason Codes"])
                    st.markdown("**Financial Health Narrative:**")
                    st.write(wr["Health Narrative"])
                else:
                    st.info("No workers to display.")
    
    # ----------------------- AML / SECURITY TAB ----------------- #
    with tab_sec:
        col_tbl_1, col_chart_5 = st.columns(2)
        
        with col_tbl_1:
            with st.container(border=True):
                st.subheader("AML & Risk Monitoring")
                base_anom = results["Baseline"]["anomalies"]
                pol_anom = results["Policy"]["anomalies"]
                
                st.markdown("#### Anomaly Table (Baseline & Policy)")
                all_anom = pd.concat([base_anom, pol_anom], ignore_index=True)
                
                if all_anom.empty:
                    st.info("No anomalies detected under current rules.")
                else:
                    st.dataframe(
                        all_anom.sort_values(["Scenario", "Emirate", "Sector"]),
                        use_container_width=True,
                        height=350,
                    )
        
        with col_chart_5:
            # Donut chart: anomalies by scenario
            with st.container(border=True):
                st.markdown("#### Anomaly Composition by Scenario")
                if all_anom.empty:
                    st.info("No anomalies to plot.")
                else:
                    scen_counts = (
                        all_anom["Scenario"]
                        .value_counts()
                        .rename_axis("Scenario")
                        .reset_index(name="Count")
                    )
                    
                    fig_donut = px.pie(
                        scen_counts,
                        names="Scenario",
                        values="Count",
                        hole=0.5,
                        color="Scenario",
                        color_discrete_map={
                            "Baseline": COLOR_BASELINE,
                            "Policy": COLOR_POLICY,
                        },
                    )
                    fig_donut.update_layout(
                        template="simple_white",
                        showlegend=True,
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
            
            # Bubble grid: anomalies by Emirate & Sector
            with st.container(border=True):
                st.markdown("#### Anomaly Grid (Emirate × Sector)")
                if all_anom.empty:
                    st.info("No anomalies to plot.")
                else:
                    heat_df = (
                        all_anom
                        .groupby(["Emirate", "Sector"])
                        .size()
                        .reset_index(name="Anomaly Count")
                    )
                    
                    fig_ac = px.scatter(
                        heat_df,
                        x="Sector",
                        y="Emirate",
                        size="Anomaly Count",
                        color="Anomaly Count",
                        color_continuous_scale="Reds",
                        size_max=40,
                    )
                    fig_ac.update_layout(
                        template="simple_white",
                        xaxis_title="Sector",
                        yaxis_title="Emirate",
                    )
                    fig_ac.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig_ac, use_container_width=True)

# FHI footer
st.caption(
    f"Simulation timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} · "
    "Baseline = today's behaviour · Policy = simulated improvement in wage stability & savings · "
    "Synthetic data only · Powered by FinYo Inclusion Engine."
)

# ====================================================================
# TAB 2 – FRI / BANK VIEW
# ====================================================================
with tab_fri:
    st.subheader("Alternative Credit Scoring – Bank Dashboard (FRI)")
    st.caption(
        "This view uses the data from The FHI to build a synthetic bank portfolio. it demonstrates how the FRI score is used to assign risk grades, "
        "calculate loan caps, and provide decision reasons."
    )
    
    if not st.session_state.get("fri_results"):
        st.info("FRI data not available. Please refresh the page.")
    else:
        fri_port = st.session_state["fri_results"]["portfolio"]
        s = st.session_state["fri_results"]["summary"]
        
        with st.container(border=True):
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1:
                st.metric("Avg FRI (0–1)", f"{s['avg_fri']:.2f}")
            with k2:
                st.metric("Avg FRI Score (0–100)", f"{s['avg_score_100']:.1f}")
            with k3:
                st.metric("Grade A + B (Low / Moderate Risk)", f"{s['count_low_mod']}")
            with k4:
                st.metric("Grade C + D (Elevated / High Risk)", f"{s['count_high']}")
            with k5:
                st.metric("Total Borrowers", f"{s['total']}")
        
        c1, c2 = st.columns(2)
        
        # --- Histogram + density line for FRI Score (0–100) ---
        with c1:
            with st.container(border=True):
                st.markdown("### FRI Score Distribution (0–100)")
                if fri_port.empty:
                    st.info("No portfolio data available.")
                else:
                    fig_hist = px.histogram(
                        fri_port,
                        x="FRI Score (0–100)",
                        nbins=20,
                        marginal=None,
                        color_discrete_sequence=[COLOR_POLICY]
                    )
                    
                    # Approximate density curve from histogram
                    scores = fri_port["FRI Score (0–100)"].dropna().values
                    if len(scores) > 1:
                        hist_y, hist_x = np.histogram(scores, bins=30, density=True)
                        x_mid = 0.5 * (hist_x[:-1] + hist_x[1:])
                        fig_hist.add_scatter(
                            x=x_mid,
                            y=hist_y,
                            mode="lines",
                            name="Density",
                        )
                    
                    fig_hist.update_layout(
                        template="simple_white",
                        xaxis_title="FRI Score (0–100)",
                        yaxis_title="Number of borrowers",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        # --- Donut Chart: Portfolio Composition by Risk Group ---
        with c2:
            with st.container(border=True):
                st.markdown("### Portfolio Composition by Risk Group")
                if fri_port.empty:
                    st.info("No portfolio data available.")
                else:
                    risk_group_df = pd.DataFrame(
                        {
                            "Risk Group": [
                                "Low / Moderate (A+B)",
                                "Elevated / High (C+D)",
                            ],
                            "Count": [
                                s["count_low_mod"],
                                s["count_high"],
                            ],
                        }
                    )
                    
                    fig_grade = px.pie(
                        risk_group_df,
                        names="Risk Group",
                        values="Count",
                        hole=0.5,
                        color="Risk Group",
                        color_discrete_map={
                            "Low / Moderate (A+B)": COLOR_POLICY,
                            "Elevated / High (C+D)": COLOR_HIGHLIGHT,
                        },
                    )
                    fig_grade.update_traces(textinfo="percent+label")
                    fig_grade.update_layout(
                        template="simple_white",
                        showlegend=True,
                        legend_title_text="Risk Group",
                        margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_grade, use_container_width=True)
        
        # --------------------------------------------------------------------
        # REPAYMENT PERFORMANCE INTELLIGENCE (Professional Layout)
        # --------------------------------------------------------------------
        with st.expander("Repayment Performance Intelligence", expanded=True):
            fri_port_display = (
                fri_port.rename(columns={"AI Reason Codes": "Reason Codes"}).copy()
                if not fri_port.empty
                else pd.DataFrame()
            )
            
            if fri_port_display.empty:
                st.info("No borrower-level data available. Run the full simulation first.")
            else:
                # --- Build compact layout ---
                left_col, right_col = st.columns([0.4, 0.6])
                
                # Left: Donut chart
                with left_col:
                    with st.container(border=True):
                        st.markdown("#### Repayment Behaviour Distribution")
                        repay_mix = (
                            fri_port_display["Repayment Behaviour"]
                            .value_counts(normalize=True)
                            .mul(100)
                            .reset_index()
                        )
                        repay_mix.columns = ["Behaviour", "Share %"]
                        
                        fig_repay = px.pie(
                            repay_mix,
                            names="Behaviour",
                            values="Share %",
                            hole=0.5,
                            color="Behaviour",
                            color_discrete_map={
                                "On-Time": COLOR_POLICY,
                                "Partial": COLOR_BASELINE,
                                "Early": COLOR_MUTED,
                                "Deferred": COLOR_HIGHLIGHT,
                            },
                        )
                        fig_repay.update_traces(textinfo="percent+label")
                        fig_repay.update_layout(
                            template="simple_white",
                            showlegend=True,
                            height=280,
                            margin=dict(t=20, b=20, l=20, r=20),
                        )
                        st.plotly_chart(fig_repay, use_container_width=True)
                
                # Right: Borrower table
                with right_col:
                    with st.container(border=True):
                        st.markdown("#### Borrower Portfolio Table")
                        display_cols = [
                            "Borrower ID",
                            "Monthly Salary (AED)",
                            "FRI Score (0–100)",
                            "Risk Grade (A–D)",
                            "Repayment Behaviour",
                            "Repayment Risk",
                            "Proactive Alert",
                            "Loan Cap (AED)",
                        ]
                        st.dataframe(
                            fri_port_display[display_cols]
                            .sort_values("FRI Score (0–100)", ascending=False),
                            use_container_width=True,
                            height=280,
                        )
        
        # Footer caption for compliance clarity
        st.caption(
            "Compact demonstration — all borrowers remain **eligible**; "
            "risk grades only adjust **loan size and duration**, not inclusion. "
            "Data shown is 100% synthetic (FinYo Inclusion Engine)."
        )
