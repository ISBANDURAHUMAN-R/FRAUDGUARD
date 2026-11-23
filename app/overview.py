"""Page 1: Overview Dashboard for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.sidebar_filters import apply_dataframe_filters
from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
    TEXT_SUBTLE,
    get_plotly_layout,
)
from src.data_loader import is_dataset_available, load_dataset
from src.predict import load_artifacts


@st.cache_data(show_spinner=False)
def get_cached_dataset():
    """Load dataset with caching to guarantee smooth interaction."""
    if not is_dataset_available():
        return None
    try:
        return load_dataset()
    except Exception:
        return None


def render_overview_page(filters: Dict[str, Any]):
    """Render the Overview Dashboard."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">EXECUTIVE OVERVIEW DASHBOARD</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Enterprise Financial Surveillance and Threat Exposure Portfolio
                </div>
            </div>
            <div class="section-tag">SURVEILLANCE ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        _, _, metadata = load_artifacts()
    except Exception:
        metadata = {}

    full_df = get_cached_dataset()
    if full_df is None:
        st.warning("European Credit Card Fraud dataset is not yet loaded in data/creditcard.csv.")
        return

    # Apply active sidebar filters
    df = apply_dataframe_filters(full_df, filters)

    total_tx = len(df)
    if total_tx == 0:
        st.info("No transactions match the selected filter criteria. Please widen your filter settings in the left sidebar.")
        return

    fraud_tx = int((df["Class"] == 1).sum()) if "Class" in df.columns else 0
    legit_tx = total_tx - fraud_tx
    fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0.0

    avg_amt = float(df["Amount"].mean()) if "Amount" in df.columns else 0.0
    max_amt = float(df["Amount"].max()) if "Amount" in df.columns else 0.0

    # Model evaluation metrics
    rf_eval = metadata.get("model_evaluations", {}).get("Random Forest", {})
    recall_pct = rf_eval.get("recall", 0.8571) * 100
    precision_pct = rf_eval.get("precision", 0.7304) * 100

    # --- SECTION 1: Top Horizontal KPI Cards ---
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

    with kpi_c1:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">TOTAL TRANSACTIONS</div>
                <div class="kpi-value">{total_tx:,}</div>
                <div class="kpi-meta">Filtered Portfolio Slice</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_c2:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">FRAUD DETECTIONS</div>
                <div class="kpi-value" style="color: {FRAUD_RED};">{fraud_tx:,}</div>
                <div class="kpi-meta" style="color: {FRAUD_RED};">Confirmed Threat Events</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_c3:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">LEGITIMATE TRANSACTIONS</div>
                <div class="kpi-value" style="color: {GREEN_PRIMARY};">{legit_tx:,}</div>
                <div class="kpi-meta" style="color: {GREEN_PRIMARY};">Cleared Volume ({(legit_tx/total_tx*100):.2f}%)</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_c4:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">FRAUD DETECTION RATE</div>
                <div class="kpi-value" style="color: {GREEN_PRIMARY};">{fraud_rate:.3f}%</div>
                <div class="kpi-meta">Anomaly Concentration</div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: Fraud Exposure & Performance Insights Row ---
    st.markdown("""
        <div style="font-size: 13px; font-weight: 700; color: #1F2937; margin: 16px 0 10px 0; letter-spacing: 0.5px; text-transform: uppercase;">
            FRAUD EXPOSURE & FINANCIAL INSIGHTS
        </div>
    """, unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    with exp_col1:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">AVERAGE AMOUNT</div>
                <div class="kpi-value" style="font-size: 22px;">€{avg_amt:,.2f}</div>
                <div class="kpi-meta">Per Transaction Mean</div>
            </div>
        """, unsafe_allow_html=True)

    with exp_col2:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">MAX TRANSACTION</div>
                <div class="kpi-value" style="font-size: 22px;">€{max_amt:,.2f}</div>
                <div class="kpi-meta">Highest Exposure Event</div>
            </div>
        """, unsafe_allow_html=True)

    with exp_col3:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">MODEL FRAUD RECALL</div>
                <div class="kpi-value" style="font-size: 22px; color: {GREEN_PRIMARY};">{recall_pct:.1f}%</div>
                <div class="kpi-meta">Threat Interception Rate</div>
            </div>
        """, unsafe_allow_html=True)

    with exp_col4:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">MODEL PRECISION</div>
                <div class="kpi-value" style="font-size: 22px; color: {GREEN_PRIMARY};">{precision_pct:.1f}%</div>
                <div class="kpi-meta">False Positive Suppression</div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECTION 3: Visual Analytics (Donut + Volume over Time) ---
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    v_col1, v_col2 = st.columns([1, 1.4])

    with v_col1:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                TRANSACTION CLASS BREAKDOWN
            </div>
        """, unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[legit_tx, fraud_tx],
            hole=0.68,
            marker=dict(colors=[GREEN_PRIMARY, FRAUD_RED]),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )])
        donut_layout = get_plotly_layout(height=300)
        fig_donut.update_layout(donut_layout)
        st.plotly_chart(fig_donut, use_container_width=True)

    with v_col2:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                FRAUD VELOCITY ACROSS RECORDING WINDOW (HOURS)
            </div>
        """, unsafe_allow_html=True)
        if "Time" in df.columns and "Class" in df.columns:
            fraud_subset = df[df["Class"] == 1]
            if not fraud_subset.empty:
                hours = (fraud_subset["Time"] / 3600.0).astype(int)
                hourly_fraud = hours.value_counts().sort_index().reset_index()
                hourly_fraud.columns = ["Hour", "Fraud_Count"]
            else:
                hourly_fraud = pd.DataFrame(columns=["Hour", "Fraud_Count"])

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=hourly_fraud["Hour"],
                y=hourly_fraud["Fraud_Count"],
                mode="lines+markers",
                name="Fraud Velocity",
                line=dict(color=FRAUD_RED, width=2.5),
                marker=dict(size=5, color=GREEN_PRIMARY),
                fill="tozeroy",
                fillcolor="rgba(220, 38, 38, 0.08)"
            ))
            time_layout = get_plotly_layout(height=300)
            time_layout["xaxis"]["title"] = "Elapsed Hours (0 - 48)"
            time_layout["yaxis"]["title"] = "Detected Fraud Events"
            fig_time.update_layout(time_layout)
            st.plotly_chart(fig_time, use_container_width=True)

    # --- SECTION 4: Amount Distribution & Recent Alerts ---
    a_col1, a_col2 = st.columns([1.2, 1])

    with a_col1:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                TRANSACTION AMOUNT SPREAD (LOG SCALE)
            </div>
        """, unsafe_allow_html=True)
        if "Amount" in df.columns and "Class" in df.columns:
            fraud_sub = df[df["Class"] == 1]["Amount"]
            legit_sub = df[df["Class"] == 0]["Amount"]
            if len(legit_sub) > 2000:
                legit_sub = legit_sub.sample(n=2000, random_state=42)

            fig_amt = go.Figure()
            fig_amt.add_trace(go.Box(
                y=np.log1p(legit_sub),
                name="Legitimate",
                marker_color=GREEN_PRIMARY,
                boxmean=True
            ))
            fig_amt.add_trace(go.Box(
                y=np.log1p(fraud_sub),
                name="Fraudulent",
                marker_color=FRAUD_RED,
                boxmean=True
            ))
            amt_layout = get_plotly_layout(height=300)
            amt_layout["yaxis"]["title"] = "Log(Amount + 1) in EUR"
            fig_amt.update_layout(amt_layout)
            st.plotly_chart(fig_amt, use_container_width=True)

    with a_col2:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                CRITICAL SURVEILLANCE FEED (TOP ANOMALIES)
            </div>
        """, unsafe_allow_html=True)
        if "Class" in df.columns:
            fraud_feed = df[df["Class"] == 1].head(5)[["Time", "Amount", "V14", "V10", "V12"]].copy()
            fraud_feed["Status"] = "INTERCEPTED"
            fraud_feed["Amount"] = fraud_feed["Amount"].map(lambda x: f"€{x:,.2f}")
            fraud_feed["Time"] = fraud_feed["Time"].map(lambda x: f"{int(x)}s")
            st.dataframe(fraud_feed, use_container_width=True, hide_index=True)
        else:
            st.info("No target labels present in current filter view.")
