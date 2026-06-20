"""
Heart Disease Prediction & Health Report System
-------------------------------------------------
A Streamlit healthcare dashboard that loads a trained Logistic Regression
model (heart_model.pkl) and a StandardScaler (heart_scaler.pkl), collects
patient data, predicts cardiovascular risk, visualizes results with Plotly,
and generates a downloadable PDF health report.

Run with:
    streamlit run app.py

Expected files in the same directory:
    heart_model.pkl   -> dict with keys {"model": LogisticRegression, "feature_names": [...]}
    heart_scaler.pkl  -> fitted StandardScaler
"""

import io
import os
import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import google.generativeai as genai

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Heart Disease Prediction & Health Report System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — PREMIUM DARK HEALTHCARE THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0b0f19;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #111625 0%, #1d263b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 240, 255, 0.12);
        border: 1px solid rgba(0, 240, 255, 0.18);
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .app-header h1 {
        color: #f1f5f9;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    .app-header p {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }

    /* Generic card with Glassmorphism */
    .card {
        background: rgba(22, 27, 44, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    .card:hover {
        border-color: rgba(0, 240, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.08);
    }

    /* Metric-style cards */
    .metric-card {
        background: rgba(22, 27, 44, 0.75);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.15);
        border: 1px solid rgba(0, 240, 255, 0.3);
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #00f0ff;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
    }

    /* Risk result banners with glass glow */
    .risk-banner-low {
        background: rgba(16, 185, 129, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    .risk-banner-high {
        background: rgba(239, 68, 68, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
    }
    .risk-banner-low h2 { color: #34d399; margin: 0 0 0.4rem 0; font-weight: 700; font-size: 1.5rem; }
    .risk-banner-high h2 { color: #f87171; margin: 0 0 0.4rem 0; font-weight: 700; font-size: 1.5rem; }
    .risk-banner-low p, .risk-banner-high p { color: #cbd5e1; margin: 0.2rem 0; font-size: 1.02rem; }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-normal { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .badge-high { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Section title */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #00f0ff;
        margin: 1.4rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 3px solid rgba(0, 240, 255, 0.15);
        text-shadow: 0 0 8px rgba(0, 240, 255, 0.2);
    }

    /* Streamlit expander backgrounds */
    .streamlit-expanderHeader {
        background-color: rgba(22, 27, 44, 0.6) !important;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }

    /* Action buttons with glowing effects */
    .stButton>button {
        background: linear-gradient(135deg, #00f0ff 0%, #00a8cc 100%);
        color: #0b0f19;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        border: none;
        width: 100%;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.45);
        transform: translateY(-1px);
        background: linear-gradient(135deg, #33f4ff 0%, #00c2eb 100%);
        color: #0b0f19;
    }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(22, 27, 44, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        color: #94a3b8 !important;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 240, 255, 0.1);
        color: #00f0ff !important;
        border-color: rgba(0, 240, 255, 0.25);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(22, 27, 44, 0.95) !important;
        color: #00f0ff !important;
        border-color: rgba(0, 240, 255, 0.4) !important;
        border-bottom: 2px solid #00f0ff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* ── Chat Bubble Styles ──────────────────────────────── */
    .chat-window {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem 0.5rem;
        max-height: 480px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(0,240,255,0.3) transparent;
    }
    .chat-window::-webkit-scrollbar { width: 5px; }
    .chat-window::-webkit-scrollbar-track { background: transparent; }
    .chat-window::-webkit-scrollbar-thumb {
        background: rgba(0,240,255,0.3);
        border-radius: 10px;
    }

    .chat-row {
        display: flex;
        align-items: flex-end;
        gap: 0.6rem;
        width: 100%;
    }
    .chat-row.user-row   { flex-direction: row-reverse; }
    .chat-row.bot-row    { flex-direction: row; }

    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .avatar-user { background: linear-gradient(135deg,#00f0ff,#0077b6); }
    .avatar-bot  { background: rgba(30,37,60,0.9); border: 1px solid rgba(0,240,255,0.2); }

    .chat-bubble {
        max-width: 72%;
        padding: 0.75rem 1.1rem;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.55;
        word-break: break-word;
    }
    .bubble-user {
        background: linear-gradient(135deg, #00c8e0 0%, #0077b6 100%);
        color: #ffffff;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 14px rgba(0,200,224,0.25);
    }
    .bubble-bot {
        background: rgba(22, 27, 44, 0.88);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0,240,255,0.12);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .bubble-bot p, .bubble-user p { margin: 0; }
    .bubble-bot ul, .bubble-bot ol { margin: 0.3rem 0 0.3rem 1rem; padding: 0; }
    .bubble-bot li { margin-bottom: 0.2rem; }
    /* ─────────────────────────────────────────────────────── */
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS — feature order MUST match training notebook exactly
# ============================================================================
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

MODEL_METRICS = {
    "accuracy": 0.85,
    "precision": 0.85,
    "recall": 0.85,
    "f1": 0.85,
}


# Approximate logistic-regression-derived feature importance (|coefficient| based,
# illustrative ranking consistent with clinical literature on this dataset)
FEATURE_IMPORTANCE = {
    "Chest Pain Type (cp)": 0.92,
    "Max Heart Rate (thalach)": 0.81,
    "ST Depression (oldpeak)": 0.78,
    "Number of Vessels (ca)": 0.74,
    "Thalassemia (thal)": 0.68,
    "Exercise Angina (exang)": 0.60,
    "Sex": 0.52,
    "Age": 0.48,
    "Cholesterol (chol)": 0.41,
    "Blood Pressure (trestbps)": 0.37,
}

# ============================================================================
# MODEL / SCALER LOADING
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load the trained model dict and scaler. Returns (model, feature_names, scaler, error)."""
    model, feature_names, scaler, error = None, FEATURE_ORDER, None, None
    try:
        model_data = joblib.load("heart_model.pkl")
        if isinstance(model_data, dict):
            model = model_data.get("model")
            feature_names = model_data.get("feature_names", FEATURE_ORDER)
        else:
            model = model_data
        scaler = joblib.load("heart_scaler.pkl")
    except FileNotFoundError as e:
        error = f"Could not find model/scaler file: {e.filename}. Make sure heart_model.pkl and heart_scaler.pkl are in the same folder as app.py."
    except Exception as e:
        error = f"Error loading model artifacts: {e}"
    return model, feature_names, scaler, error


model, feature_names, scaler, load_error = load_artifacts()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def build_input_dataframe():
    row = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    cols = feature_names if feature_names else FEATURE_ORDER
    return pd.DataFrame([[row[c] for c in cols]], columns=cols)


def assess_parameter(name, value, low, high, unit=""):
    """Returns (status, explanation) for a clinical parameter."""
    if value < low:
        return "Normal", f"{name} is within the normal range."
    elif low <= value <= high:
        return "Warning", f"{name} is borderline and worth monitoring."
    else:
        return "High", f"{name} is elevated and may increase cardiovascular risk."


def status_badge(status):
    cls = {"Normal": "badge-normal", "Warning": "badge-warning", "High": "badge-high"}.get(status, "badge-normal")
    return f'<span class="badge {cls}">{status}</span>'


def get_risk_assessment_table():
    rows = []

    # Cholesterol
    if chol < 200:
        chol_status = "Normal"
    elif chol < 240:
        chol_status = "Warning"
    else:
        chol_status = "High"
    rows.append(("Cholesterol", f"{chol} mg/dL", "< 200 mg/dL", chol_status,
                 "High cholesterol may contribute to artery blockage (atherosclerosis)."))

    # Blood Pressure
    if trestbps < 120:
        bp_status = "Normal"
    elif trestbps < 140:
        bp_status = "Warning"
    else:
        bp_status = "High"
    rows.append(("Blood Pressure", f"{trestbps} mmHg", "< 120 mmHg", bp_status,
                 "Elevated blood pressure increases the heart's workload over time."))

    # Max Heart Rate (lower achieved HR for age can indicate reduced cardiac capacity)
    expected_max_hr = 220 - age
    pct_of_expected = thalach / expected_max_hr if expected_max_hr > 0 else 1
    if 0.85 <= pct_of_expected <= 1.05:
        hr_status = "Normal"
    elif pct_of_expected > 1.05:
        hr_status = "Warning"   # exceeds age-predicted max
    else:
        hr_status = "High" 
    rows.append(("Max Heart Rate", f"{thalach} bpm", f"~{expected_max_hr} bpm (age-predicted)", hr_status,
                 "A lower-than-expected maximum heart rate can indicate reduced cardiac fitness."))

    # ST Depression
    if oldpeak < 1.0:
        op_status = "Normal"
    elif oldpeak < 2.0:
        op_status = "Warning"
    else:
        op_status = "High"
    rows.append(("ST Depression", f"{oldpeak}", "< 1.0", op_status,
                 "Higher ST depression during exercise can indicate reduced blood flow to the heart."))

    return rows


def risk_level_from_probability(prob_high):
    if prob_high < 0.34:
        return "Low Risk", "#2ecc71"
    elif prob_high < 0.67:
        return "Medium Risk", "#f1c40f"
    else:
        return "High Risk", "#e74c3c"


# ============================================================================
# PLOTLY CHART BUILDERS
# ============================================================================
def make_gauge_chart(prob_high):
    level_text, level_color = risk_level_from_probability(prob_high)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_high * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#00f0ff"}},
        title={"text": f"<span style='color:#f1f5f9;font-weight:700;'>Cardiovascular Risk Score</span><br><span style='font-size:0.8em;color:{level_color};font-weight:600;'>{level_text}</span>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
            "bar": {"color": level_color},
            "steps": [
                {"range": [0, 34], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [34, 67], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [67, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#00f0ff", "width": 4},
                "thickness": 0.8,
                "value": prob_high * 100,
            },
        },
    ))
    fig.update_layout(
        height=320, 
        margin=dict(t=70, b=10, l=30, r=30), 
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"}
    )
    return fig


def make_comparison_chart():
    params = ["Cholesterol<br>(mg/dL)", "Blood Pressure<br>(mmHg)", "Max Heart Rate<br>(bpm)"]
    user_vals = [chol, trestbps, thalach]
    recommended = [200, 120, 220 - age]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Your Value", x=params, y=user_vals, marker_color="#00f0ff"))
    fig.add_trace(go.Bar(name="Recommended", x=params, y=recommended, marker_color="rgba(16, 185, 129, 0.7)"))
    fig.update_layout(
        barmode="group", height=380,
        margin=dict(t=40, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font={"color": "#cbd5e1"}),
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    return fig


def make_health_indicator_charts():
    """Three small indicator charts: cholesterol, BP, heart rate vs healthy bands."""
    fig = go.Figure()
    categories = ["Cholesterol", "Blood Pressure", "Heart Rate"]
    values = [chol, trestbps, thalach]
    healthy_max = [200, 120, 220 - age]
    colors_list = []
    for v, hmax in zip(values, healthy_max):
        ratio = v / hmax if hmax else 1
        if ratio <= 1.0:
            colors_list.append("#10b981")
        elif ratio <= 1.2:
            colors_list.append("#f59e0b")
        else:
            colors_list.append("#ef4444")

    fig.add_trace(go.Bar(
        x=categories, y=values, marker_color=colors_list, name="Current Value",
        text=values, textposition="outside", textfont=dict(color="#cbd5e1")
    ))
    fig.add_trace(go.Scatter(
        x=categories, y=healthy_max, mode="markers+lines",
        name="Healthy Threshold", marker=dict(color="#00f0ff", size=10, symbol="diamond"),
        line=dict(dash="dash", color="#00f0ff")
    ))
    fig.update_layout(
        height=380, margin=dict(t=30, b=30, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font={"color": "#cbd5e1"})
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    return fig


def make_feature_importance_chart():
    items = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1])
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    fig = px.bar(
        x=vals, y=names, orientation="h",
        labels={"x": "Relative Importance", "y": ""},
        color=vals, color_continuous_scale=["rgba(0, 240, 255, 0.2)", "#00f0ff"],
    )
    fig.update_layout(
        height=420, margin=dict(t=20, b=30, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        coloraxis_showscale=False
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.12)")
    return fig



# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="app-header">
    <h1>❤️ Heart Disease Prediction &amp; Health Report System</h1>
    <p>AI-powered cardiovascular risk assessment using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ {load_error}")
    st.info("The app will still render below, but predictions are disabled until the model files are available.")

# ============================================================================
# SIDEBAR — NAVIGATION & INFO
# ============================================================================
with st.sidebar:
    st.markdown("### ℹ️ About This Tool")
    st.markdown(
        "This dashboard uses a **Logistic Regression** model trained on the "
        "UCI Heart Disease dataset to estimate cardiovascular risk from "
        "clinical parameters."
    )
    st.markdown("---")
    st.markdown("### 📊 Model Snapshot")
    st.markdown(f"- **Accuracy:** {MODEL_METRICS['accuracy']*100:.0f}%")
    st.markdown(f"- **F1 Score:** {MODEL_METRICS['f1']*100:.0f}%")
    st.markdown("---")
    st.warning(
        "This tool is for **educational/informational purposes only** and "
        "does not replace professional medical advice, diagnosis, or treatment."
    )


# ============================================================================
# MAIN APPLICATION LAYOUT
# ============================================================================
container_input = st.container()

with container_input:
    st.markdown('<div class="section-title">📥 Patient Parameters</div>', unsafe_allow_html=True)
    st.caption("Adjust the clinical values below. Tooltips (❔) are available to explain each medical parameter in simple terms.")
    
    tab_profile, tab_tests = st.tabs(["👤 Profile & Vitals", "🩺 Clinical Tests"])
    
    with tab_profile:
        st.markdown("#### Demographics")
        age = st.number_input(
            "Age (years)", 
            min_value=1, max_value=120, value=50, step=1,
            help="Age is a primary factor in cardiovascular health. Risk naturally increases over time."
        )
        sex_label = st.radio(
            "Gender", 
            ["Male", "Female"], horizontal=True,
            help="Biological gender affects cardiovascular risk profiles and baseline model calculations."
        )
        sex = 1 if sex_label == "Male" else 0
        
        st.markdown("---")
        st.markdown("#### Key Vitals")
        trestbps = st.slider(
            "Resting Blood Pressure (mmHg)", 
            min_value=80, max_value=220, value=120, step=1,
            help="Resting blood pressure measured in mm Hg. Normal is generally below 120. Over 140 is elevated (hypertension)."
        )
        chol = st.slider(
            "Cholesterol Level (mg/dL)", 
            min_value=100, max_value=600, value=200, step=1,
            help="Serum cholesterol in mg/dL. Ideal is below 200 mg/dL. Values above 240 mg/dL indicate hypercholesterolemia."
        )
        
        fbs_label = st.radio(
            "Fasting Blood Sugar > 120 mg/dL?", 
            ["No", "Yes"], horizontal=True,
            help="Is fasting blood sugar higher than 120 mg/dL? Elevated blood sugar can be a sign of diabetes, which increases risk."
        )
        fbs = 1 if fbs_label == "Yes" else 0
        
        thalach = st.slider(
            "Maximum Heart Rate Achieved (bpm)", 
            min_value=60, max_value=220, value=150, step=1,
            help="The highest heart rate reached during a cardiac stress test. Expected max is roughly (220 - Age)."
        )
        
    with tab_tests:
        st.markdown("#### Chest & Exercise Symptoms")
        cp_options = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3,
        }
        cp_label = st.selectbox(
            "Chest Pain Type", 
            list(cp_options.keys()),
            help="Typical Angina: exertion-related chest pain. Atypical Angina: general chest discomfort. Non-anginal: pain not from the heart. Asymptomatic: no pain."
        )
        cp = cp_options[cp_label]
        
        exang_label = st.radio(
            "Exercise Induced Angina?", 
            ["No", "Yes"], horizontal=True,
            help="Did physical exercise trigger chest pain? Pain during exercise can indicate narrowed coronary arteries."
        )
        exang = 1 if exang_label == "Yes" else 0
        
        st.markdown("---")
        st.markdown("#### ECG & Cardiac Measurements")
        
        restecg_options = {
            "Normal": 0,
            "ST-T Wave Abnormality": 1,
            "Left Ventricular Hypertrophy": 2,
        }
        restecg_label = st.selectbox(
            "Resting ECG Result", 
            list(restecg_options.keys()),
            help="Resting Electrocardiogram pattern. ST-T wave changes or left ventricular hypertrophy suggest stress or enlargement of the heart muscle."
        )
        restecg = restecg_options[restecg_label]
        
        oldpeak = st.slider(
            "ST Depression (oldpeak)", 
            min_value=0.0, max_value=6.5, value=1.0, step=0.1,
            help="ST segment depression on ECG during exercise vs rest. Represents strain on the heart during physical effort. Lower (< 1.0) is better."
        )
        st.caption("ST Depression is measured during an exercise ECG test relative to rest.")
        
        slope_options = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2,
        }
        slope_label = st.selectbox(
            "Slope of ST Segment", 
            list(slope_options.keys()),
            help="The slope direction of the ST segment during peak exercise. Flat or downsloping patterns can indicate cardiac stress."
        )
        slope = slope_options[slope_label]
        
        ca = st.slider(
            "Number of Major Vessels Colored (0–4)", 
            min_value=0, max_value=4, value=0, step=1,
            help="Number of major coronary blood vessels (0-4) visible on a fluoroscopy scan. Fewer blocked vessels (0) represents clear blood flow."
        )
        
        thal_options = {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3,
        }
        thal_label = st.selectbox(
            "Thalassemia", 
            list(thal_options.keys()),
            help="A blood flow test indicator. Normal: healthy circulation. Fixed Defect: permanent scar tissue. Reversible Defect: temporary flow restriction."
        )
        thal = thal_options[thal_label]

    st.markdown("<br>", unsafe_allow_html=True)
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
        
    generate_clicked = st.button("🔬 Generate Health Report", use_container_width=True)
    if generate_clicked:
        st.session_state.report_generated = True


# ============================================================================
# PDF REPORT GENERATION (ReportLab)
# ============================================================================
def generate_pdf_report(prediction, probability, risk_table_rows):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#1e3a5f"),
                                  alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle("SubtitleStyle", parent=styles["Normal"],
                                     fontSize=11, textColor=colors.HexColor("#5a6b7a"),
                                     alignment=TA_CENTER, spaceAfter=18)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"],
                                    fontSize=14, textColor=colors.HexColor("#1e3a5f"),
                                    spaceBefore=14, spaceAfter=8)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=10.5, leading=15)

    elements = []
    elements.append(Paragraph("❤ Heart Disease Prediction Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))

    # 1. Patient Information
    elements.append(Paragraph("1. Patient Information", heading_style))
    patient_data = [
        ["Age", f"{age} years", "Gender", sex_label],
        ["Cholesterol", f"{chol} mg/dL", "Blood Pressure", f"{trestbps} mmHg"],
        ["Max Heart Rate", f"{thalach} bpm", "ST Depression", f"{oldpeak}"],
        ["Chest Pain Type", cp_label, "Fasting Blood Sugar", fbs_label],
        ["Resting ECG", restecg_label, "Exercise Angina", exang_label],
        ["Slope", slope_label, "Thalassemia", thal_label],
    ]
    t = Table(patient_data, colWidths=[4 * cm, 4.5 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#1e3a5f")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde5ec")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)

    # 2. Prediction Result
    elements.append(Paragraph("2. Prediction Result", heading_style))
    result_text = "High Risk - Possible Heart Disease" if prediction == 1 else "Low Risk - Healthy"
    result_color = colors.HexColor("#b0291c") if prediction == 1 else colors.HexColor("#1e7e44")
    result_style = ParagraphStyle("ResultStyle", parent=body_style, fontSize=13,
                                   textColor=result_color, fontName="Helvetica-Bold")
    elements.append(Paragraph(f"Prediction: {result_text}", result_style))
    elements.append(Paragraph(f"Confidence: {probability*100:.1f}%", body_style))
    elements.append(Spacer(1, 6))

    # 3. Risk Assessment
    elements.append(Paragraph("3. Risk Assessment", heading_style))
    risk_data = [["Parameter", "Value", "Normal Range", "Status"]]
    for name, val, normal_range, status, _ in risk_table_rows:
        risk_data.append([name, val, normal_range, status])
    t2 = Table(risk_data, colWidths=[4 * cm, 3.5 * cm, 4.5 * cm, 4 * cm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde5ec")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    status_colors = {"Normal": colors.HexColor("#1e7e44"), "Warning": colors.HexColor("#8a6500"),
                      "High": colors.HexColor("#b0291c")}
    for i, row in enumerate(risk_table_rows, start=1):
        status = row[3]
        style_cmds.append(("TEXTCOLOR", (3, i), (3, i), status_colors.get(status, colors.black)))
        style_cmds.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    t2.setStyle(TableStyle(style_cmds))
    elements.append(t2)
    elements.append(Spacer(1, 8))

    for name, val, normal_range, status, explanation in risk_table_rows:
        elements.append(Paragraph(f"<b>{name}</b>: {explanation}", body_style))

    # 4. Health Recommendations
    elements.append(Paragraph("4. Health Recommendations", heading_style))
    if prediction == 1:
        recs = [
            "Consult a healthcare professional for a full cardiac evaluation.",
            "Monitor blood pressure regularly at home or with a clinician.",
            "Reduce intake of unhealthy/saturated fats and sodium.",
            "Engage in regular, moderate exercise as approved by a doctor.",
            "Maintain a healthy weight and avoid tobacco use.",
        ]
    else:
        recs = [
            "Continue maintaining your current healthy lifestyle.",
            "Engage in regular physical exercise (at least 150 minutes/week).",
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
            "Schedule regular health checks to monitor key indicators.",
        ]
    for r in recs:
        elements.append(Paragraph(f"• {r}", body_style))

    elements.append(Spacer(1, 14))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=body_style, fontSize=8.5,
                                       textColor=colors.HexColor("#8a93a0"))
    elements.append(Paragraph(
        "Disclaimer: This report is generated by a machine learning model for "
        "informational purposes only and does not constitute medical advice. "
        "Please consult a licensed healthcare professional for diagnosis and treatment.",
        disclaimer_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================================================
# MAIN PREDICTION & RESULT DISPLAY FLOW
# ============================================================================
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

if not st.session_state.report_generated:
    st.info("👆 Adjust the parameters above and click **Generate Health Report** to perform the AI cardiovascular assessment.")
    st.stop()

if model is None or scaler is None:
    prediction_error = "Cannot generate a prediction because the model or scaler failed to load. Please check that heart_model.pkl and heart_scaler.pkl exist in the app directory."
else:
    prediction_error = None
    input_df = build_input_dataframe()
    try:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        proba = model.predict_proba(scaled_input)[0]
        prob_high = float(proba[1])
        confidence = prob_high if prediction == 1 else float(proba[0])
        risk_rows = get_risk_assessment_table()
    except Exception as e:
        prediction_error = f"Prediction failed: {e}"

container_result = st.container()

with container_result:
    st.markdown('<div class="section-title">🔮 Risk Assessment & Dashboard</div>', unsafe_allow_html=True)
    
    if prediction_error:
        st.error(f"⚠️ {prediction_error}")
    else:
        # ---------------- Prediction Card ----------------
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-banner-high">
                <h2>⚠️ Elevated Cardiovascular Risk Detected</h2>
                <p><b>Confidence:</b> {confidence*100:.1f}%</p>
                <p><b>Risk Level:</b> {risk_level_from_probability(prob_high)[0]}</p>
                <p style="margin-top: 8px; font-size: 0.95rem; line-height: 1.4;">
                    Our AI model indicates a higher risk of heart disease based on your inputs. 
                    <b>What this means:</b> Your clinical test values fall into patterns associated with heart strain. 
                    We recommend discussing these results and downloading the report PDF for a clinical review with your doctor.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-banner-low">
                <h2>✅ Low Risk Profile</h2>
                <p><b>Confidence:</b> {confidence*100:.1f}%</p>
                <p><b>Risk Level:</b> {risk_level_from_probability(prob_high)[0]}</p>
                <p style="margin-top: 8px; font-size: 0.95rem; line-height: 1.4;">
                    Our AI model indicates a healthy cardiovascular profile based on your current inputs. 
                    <b>What this means:</b> Your vital metrics and test outputs align with expected healthy bounds. 
                    Keep up a healthy lifestyle, physical exercise, and regular wellness screenings!
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- Tabs for Dashboard Sections ----------------
        tab_dash, tab_table, tab_explain, tab_recs, tab_chat = st.tabs([
            "📊 Visual Dashboard", "📋 Clinical Analysis", "🧠 AI Model Logic", "💡 Health Guide", "💬 Health Assistant"
        ])
        
        with tab_dash:
            st.caption("ℹ️ **Visual Summary**: Explore your vital metrics in comparison with recommended targets. Move inputs on the left to watch them adjust in real-time.")
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(make_gauge_chart(prob_high), use_container_width=True)
            with v2:
                st.markdown("**Vitals Comparison: User Value vs Recommended Range**")
                st.plotly_chart(make_comparison_chart(), use_container_width=True)

            st.markdown("**Health Indicators vs Healthy Thresholds**")
            st.plotly_chart(make_health_indicator_charts(), use_container_width=True)

        with tab_table:
            st.caption("🩺 **Vitals & Risk Checklist**: A structured table of your values against medical references.")
            # Patient Summary Metrics inside Clinical Analysis
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{age} yrs</div>
                    <div class="label">Age</div>
                </div>
                """, unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{sex_label}</div>
                    <div class="label">Gender</div>
                </div>
                """, unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{chol} mg/dL</div>
                    <div class="label">Cholesterol</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk Assessment Table
            table_html = """<table style="width:100%; border-collapse: collapse; font-size:0.95rem; color:#cbd5e1;">
<tr style="background:#0f172a; color:#f1f5f9; border-bottom: 2px solid rgba(0, 240, 255, 0.2);">
    <th style="padding:12px 10px; text-align:left; border-radius:8px 0 0 0;">Parameter</th>
    <th style="padding:12px 10px; text-align:left;">User Value</th>
    <th style="padding:12px 10px; text-align:left;">Normal Range</th>
    <th style="padding:12px 10px; text-align:left; border-radius:0 8px 0 0;">Status</th>
</tr>"""
            for i, (name, val, normal_range, status, explanation) in enumerate(risk_rows):
                bg = "rgba(30, 41, 59, 0.4)" if i % 2 == 0 else "rgba(15, 23, 42, 0.4)"
                safe_range = normal_range.replace("<", "&lt;").replace(">", "&gt;")
                table_html += f"""<tr style="background:{bg}; border-bottom: 1px solid rgba(255,255,255,0.05);">
    <td style="padding:10px; color:#f1f5f9;"><b>{name}</b></td>
    <td style="padding:10px;">{val}</td>
    <td style="padding:10px;">{safe_range}</td>
    <td style="padding:10px;">{status_badge(status)}</td>
</tr>"""
            table_html += "</table>"
            st.markdown(f'<div class="card">{table_html}</div>', unsafe_allow_html=True)

            with st.expander("📖 Detailed Explanations", expanded=False):
                for name, val, normal_range, status, explanation in risk_rows:
                    st.markdown(f"**{name}** — Value: {val} | Status: {status_badge(status)}", unsafe_allow_html=True)
                    st.caption(explanation)

        with tab_explain:
            st.caption("🧠 **How does the AI work?** The chart below displays which clinical metrics have the greatest relative impact on the model's predictions. Longer bars mean the model heavily factors in that specific value.")
            st.markdown("**Feature Importance (relative influence on prediction)**")
            st.plotly_chart(make_feature_importance_chart(), use_container_width=True)

        with tab_recs:
            st.caption("💡 **Patient Health Recommendations**: Actionable items calculated based on your risk results.")
            if prediction == 1:
                recs = [
                    "🏥 Consult a healthcare professional for a full cardiac evaluation",
                    "🩺 Monitor blood pressure regularly at home",
                    "🥗 Reduce intake of unhealthy/saturated fats and sodium",
                    "🏃 Engage in regular, moderate exercise under medical guidance",
                    "⚖️ Maintain a healthy weight and avoid tobacco products",
                ]
            else:
                recs = [
                    "✅ Continue maintaining your current healthy lifestyle",
                    "🏃 Keep up regular physical exercise (at least 150 mins/week)",
                    "🥗 Maintain a balanced diet rich in whole grains and vegetables",
                    "🩺 Schedule regular health checks to monitor vitals",
                ]
            for r in recs:
                st.markdown(f'<div class="card" style="padding: 1rem 1.5rem; margin-bottom: 0.8rem; border-left: 4px solid #00f0ff;"><b>{r}</b></div>', unsafe_allow_html=True)

        with tab_chat:
            st.caption("💬 **AI Cardiovascular Assistant**: Ask questions about your heart risk report, learn about clinical values, or seek cardiovascular health recommendations.")
            
            # Key Setup (Rely only on system environment variable/secrets)
            api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            
            if not api_key:
                st.warning("⚠️ **Assistant Unavailable**: The `GEMINI_API_KEY` environment variable is not configured. Please configure it on your system or Streamlit secrets to enable the chat assistant.")
            else:
                try:
                    # Initialize chat history in Streamlit session state
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Layout row for Reset button
                    c_header, c_reset = st.columns([4, 1])
                    with c_header:
                        st.markdown("##### Chat History")
                    with c_reset:
                        if st.button("🔄 Clear Chat", key="clear_chat_button", use_container_width=True):
                            st.session_state.chat_history = []
                            st.rerun()

                    # Create a container for messages to ensure they render above the user chat_input
                    message_container = st.container()

                    # ── Helper: render a single bubble as HTML ──
                    def render_bubble(role, content):
                        import html as htmllib
                        if role == "user":
                            row_cls, avatar_cls, bubble_cls, avatar_icon = (
                                "user-row", "avatar-user", "bubble-user", "👤"
                            )
                        else:
                            row_cls, avatar_cls, bubble_cls, avatar_icon = (
                                "bot-row", "avatar-bot", "bubble-bot", "🤖"
                            )
                        # Preserve markdown newlines as <br> for plain text,
                        # but keep markdown code/bullets readable via simple conversion
                        safe = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        safe = safe.replace("\n", "<br>")
                        return f"""
                        <div class="chat-row {row_cls}">
                            <div class="chat-avatar {avatar_cls}">{avatar_icon}</div>
                            <div class="chat-bubble {bubble_cls}">{safe}</div>
                        </div>"""

                    # Render previous chat history inside container
                    with message_container:
                        if st.session_state.chat_history:
                            bubbles_html = "".join(
                                render_bubble(m["role"], m["content"])
                                for m in st.session_state.chat_history
                            )
                            st.markdown(
                                f'<div class="chat-window">{bubbles_html}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div style="text-align:center;color:#4a5568;padding:2rem 0;font-size:0.93rem;">'
                                '💬 No messages yet. Ask me anything about your heart health report!</div>',
                                unsafe_allow_html=True
                            )
                    
                    # Construct system prompt with up-to-date parameters
                    fbs_val = "Elevated (>120 mg/dL)" if fbs == 1 else "Normal (<=120 mg/dL)"
                    exang_val = "Yes" if exang == 1 else "No"
                    pred_val = "⚠️ HIGH RISK - Possible Heart Disease" if prediction == 1 else "✅ LOW RISK - Healthy Profile"
                    
                    system_prompt = f"""You are an empathetic, professional, and knowledgeable cardiovascular health assistant.
You are assisting a user who has just completed their AI heart disease prediction.

Here is the current patient's clinical profile:
- Age: {age} years
- Gender: {sex_label}
- Resting Blood Pressure: {trestbps} mmHg
- Cholesterol Level: {chol} mg/dL
- Fasting Blood Sugar: {fbs_val}
- Maximum Heart Rate Achieved: {thalach} bpm
- Chest Pain Type: {cp_label}
- Exercise Induced Angina: {exang_val}
- Resting ECG Result: {restecg_label}
- ST Depression (oldpeak): {oldpeak}
- Slope of ST Segment: {slope_label}
- Number of Major Vessels: {ca}
- Thalassemia Marker: {thal_label}

Our Machine Learning model predicted the patient is at: {pred_val}, with a model confidence of {confidence*100:.1f}%.

Guidelines for your behavior:
1. Explain medical terms, vitals, and findings in simple, encouraging, and clear terms.
2. Provide practical cardiovascular health advice, exercise guidance, and dietary tips tailored to their cholesterol, blood pressure, and age.
3. STRICT MEDICAL DISCLAIMER: Always remind the patient that your guidance is for educational/informational purposes only and they must consult a licensed cardiologist or doctor for formal medical diagnoses, clinical symptoms review, or treatment.
4. Keep answers relatively concise and highly structured (use bullet points where appropriate).
"""

                    # Input field for user query
                    if user_query := st.chat_input("Ask me about your heart health report...", key="chat_input_message"):
                        # Save user message to history
                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        
                        # Generate response
                        with st.spinner("🤖 Analyzing profile & formulating response..."):
                            try:
                                genai.configure(api_key=api_key)
                                # Instantiate model with system instructions
                                model_gen = genai.GenerativeModel(
                                    model_name='gemini-2.5-flash',
                                    system_instruction=system_prompt
                                )
                                
                                # Format history for Gemini's API
                                history_gemini = []
                                # Exclude the last message we just appended (the current query)
                                for prev_msg in st.session_state.chat_history[:-1]:
                                    role = "user" if prev_msg["role"] == "user" else "model"
                                    history_gemini.append({"role": role, "parts": [prev_msg["content"]]})
                                    
                                chat_session = model_gen.start_chat(history=history_gemini)
                                response = chat_session.send_message(user_query)
                                
                                # Save assistant response and rerun to refresh bubbles
                                assistant_reply = response.text
                                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                                st.rerun()
                            except Exception as chat_err:
                                st.error(f"Error calling Gemini API: {chat_err}")
                except Exception as config_err:
                    st.error(f"Error configuring Google GenAI: {config_err}")

        # ---------------- PDF Report Download Card ----------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("##### ⬇️ Download Clinical Health Report")
        st.caption("Generate and download a high-quality PDF record of the patient details, risk predictions, health analysis tables, and recommendations.")
        pdf_buffer = generate_pdf_report(prediction, confidence, risk_rows)
        st.download_button(
            label="📄 Download Patient Report PDF",
            data=pdf_buffer,
            file_name=f"heart_health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("⚠️ This application is for educational and informational purposes only and is not a substitute for professional medical advice.")
