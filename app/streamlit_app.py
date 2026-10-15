from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = str(ROOT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.models.predict import SegmentPredictor

st.set_page_config(
    page_title="Customer Segmentation & RFM Profiler",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return SegmentPredictor(str(ROOT_DIR / "models" / "model_v1.joblib"))


predictor = load_predictor()

st.title("🛍️ Retail Customer RFM Segmentation & Persona Service")
st.markdown(
    "Unsupervised machine learning customer segmentation on the Online Retail II cohort (5,878 customers) using Recency, Frequency, Monetary (RFM) feature engineering and K-Means clustering."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Analyzed Customers", "5,878")
col2.metric("Total Cohort Spend", "£18.04M")
col3.metric("Optimal Clusters (K)", "5")
col4.metric("K-Means Silhouette", "0.365")

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Customer Transaction Profile (RFM)")

    preset = st.selectbox(
        "Load Customer Archetype Preset",
        [
            "Custom",
            "Champion (Elite Spender)",
            "Loyal Customer (Regular Buyer)",
            "Potential Loyalist (Recent Onboard)",
            "At Risk (Slipping Away)",
            "Hibernating (Inactive Churn)",
        ],
    )

    if preset == "Champion (Elite Spender)":
        init_r, init_f, init_m = 15.0, 35, 18000.0
    elif preset == "Loyal Customer (Regular Buyer)":
        init_r, init_f, init_m = 45.0, 10, 3500.0
    elif preset == "Potential Loyalist (Recent Onboard)":
        init_r, init_f, init_m = 25.0, 3, 700.0
    elif preset == "At Risk (Slipping Away)":
        init_r, init_f, init_m = 280.0, 4, 1500.0
    elif preset == "Hibernating (Inactive Churn)":
        init_r, init_f, init_m = 420.0, 1, 200.0
    else:
        init_r, init_f, init_m = 30.0, 5, 1200.0

    recency = st.slider(
        "Recency (Days Since Last Purchase)",
        min_value=0.0,
        max_value=730.0,
        value=float(init_r),
        step=1.0,
    )
    frequency = st.number_input(
        "Frequency (Total Distinct Invoices / Orders)",
        min_value=1,
        max_value=300,
        value=int(init_f),
        step=1,
    )
    monetary = st.number_input(
        "Monetary Value (Total Spend in £)",
        min_value=0.0,
        max_value=250000.0,
        value=float(init_m),
        step=50.0,
    )

with col_right:
    st.subheader("Segmentation Outcome & Marketing Action")
    result = predictor.predict(
        recency=recency, frequency=frequency, monetary=monetary
    )

    persona = result["persona"]
    cluster_id = result["cluster"]

    color_map = {
        "Champions": "#1b5e20",
        "Loyal Customers": "#0d47a1",
        "Potential Loyalists": "#006064",
        "At Risk": "#e65100",
        "Hibernating": "#b71c1c",
    }
    badge_color = color_map.get(persona, "#333333")

    st.markdown(
        f"""
        <div style="background-color: {badge_color}; padding: 18px; border-radius: 8px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0; color: white;">{persona}</h2>
            <p style="margin:5px 0 0 0; font-size: 16px;">Assigned Cluster ID: {cluster_id}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Customer Behavior Profile:**\n{result['description']}")
    st.markdown(
        f"**Recommended Strategic Intervention:**\n{result['recommended_action']}"
    )

    st.markdown("### Feature Benchmark vs Cohort Averages")
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Recency", f"{recency:.0f} d", delta=f"{recency - 200.4:.0f} d vs avg", delta_color="inverse")
    c_m2.metric("Frequency", f"{frequency} orders", delta=f"{frequency - 5.8:.1f} vs avg")
    c_m3.metric("Monetary", f"£{monetary:,.0f}", delta=f"£{monetary - 3069.1:,.0f} vs avg")

st.divider()
st.subheader("Global Customer Cohort Profiles (5,878 Customers Analyzed)")

profiles_data = [
    {
        "Persona": "Champions",
        "Cluster": 3,
        "Customers": 462,
        "Share (%)": "7.86%",
        "Avg Recency": "18.2 d",
        "Avg Frequency": "32.8",
        "Avg Spend": "£21,171.50",
        "Total Revenue": "£9,781,232.82",
        "Rev Share": "54.21%",
    },
    {
        "Persona": "Loyal Customers",
        "Cluster": 1,
        "Customers": 1295,
        "Share (%)": "22.03%",
        "Avg Recency": "48.9 d",
        "Avg Frequency": "9.1",
        "Avg Spend": "£3,721.82",
        "Total Revenue": "£4,819,756.01",
        "Rev Share": "26.71%",
    },
    {
        "Persona": "Potential Loyalists",
        "Cluster": 4,
        "Customers": 1141,
        "Share (%)": "19.41%",
        "Avg Recency": "32.7 d",
        "Avg Frequency": "2.5",
        "Avg Spend": "£688.30",
        "Total Revenue": "£785,351.08",
        "Rev Share": "4.35%",
    },
    {
        "Persona": "At Risk",
        "Cluster": 2,
        "Customers": 1364,
        "Share (%)": "23.21%",
        "Avg Recency": "291.6 d",
        "Avg Frequency": "3.8",
        "Avg Spend": "£1,415.52",
        "Total Revenue": "£1,930,770.29",
        "Rev Share": "10.70%",
    },
    {
        "Persona": "Hibernating",
        "Cluster": 0,
        "Customers": 1616,
        "Share (%)": "27.49%",
        "Avg Recency": "418.7 d",
        "Avg Frequency": "1.2",
        "Avg Spend": "£263.81",
        "Total Revenue": "£426,318.97",
        "Rev Share": "2.36%",
    },
]

st.dataframe(pd.DataFrame(profiles_data), use_container_width=True)
