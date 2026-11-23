"""Page 2: In-Depth Transaction & Dataset Analysis for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.sidebar_filters import apply_dataframe_filters
from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
    get_plotly_layout,
)
from src.data_loader import is_dataset_available, load_dataset


@st.cache_data(show_spinner=False)
def load_analysis_dataset():
    """Cached loader for deep transaction analysis."""
    if not is_dataset_available():
        return None
    try:
        return load_dataset()
    except Exception:
        return None


def render_transaction_analysis_page(filters: Dict[str, Any]):
    """Render comprehensive transaction dimensions and statistical breakdown."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">TRANSACTION BEHAVIORAL ANALYTICS</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Empirical Cohort Slicing, Amount Tier Risk Profiles, and Statistical Dimensionality
                </div>
            </div>
            <div class="section-tag">DATASET AUDIT</div>
        </div>
    """, unsafe_allow_html=True)

    full_df = load_analysis_dataset()
    if full_df is None:
        st.warning("European Credit Card Fraud dataset is unavailable.")
        return

    df = apply_dataframe_filters(full_df, filters)
    total_tx = len(df)
    if total_tx == 0:
        st.info("No transaction data matches the active filter criteria.")
        return

    # --- SECTION 1: Summary Statistics Cards ---
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">ANALYZED VOLUME</div>
                <div class="kpi-value">{total_tx:,}</div>
                <div class="kpi-meta">Filtered Records</div>
            </div>
        """, unsafe_allow_html=True)

    with s_col2:
        fraud_cases = int((df["Class"] == 1).sum()) if "Class" in df.columns else 0
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">DETECTED FRAUD CASES</div>
                <div class="kpi-value" style="color: {FRAUD_RED};">{fraud_cases:,}</div>
                <div class="kpi-meta" style="color: {FRAUD_RED};">Flagged Threats</div>
            </div>
        """, unsafe_allow_html=True)

    with s_col3:
        avg_amt = float(df["Amount"].mean()) if "Amount" in df.columns else 0.0
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">AVERAGE AMOUNT</div>
                <div class="kpi-value" style="color: {GREEN_PRIMARY};">€{avg_amt:,.2f}</div>
                <div class="kpi-meta">Mean Transaction Value</div>
            </div>
        """, unsafe_allow_html=True)

    with s_col4:
        total_exposure = float(df[df["Class"] == 1]["Amount"].sum()) if "Class" in df.columns else 0.0
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">FRAUD CAPITAL AT RISK</div>
                <div class="kpi-value" style="color: {FRAUD_RED};">€{total_exposure:,.2f}</div>
                <div class="kpi-meta">Total Fraud Volume</div>
            </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: Monetary Amount Tier Risk Breakdown ---
    st.markdown("""
        <div style="font-size: 13px; font-weight: 700; color: #1F2937; margin: 16px 0 10px 0; letter-spacing: 0.5px; text-transform: uppercase;">
            MONETARY AMOUNT BRACKET BREAKDOWN
        </div>
    """, unsafe_allow_html=True)

    bins = [-1.0, 10.0, 50.0, 200.0, 1000.0, float("inf")]
    labels = [
        "Micro-Transactions (< €10)",
        "Daily Retail (€10 - €50)",
        "Mid-Range (€50 - €200)",
        "High Value (€200 - €1,000)",
        "Extreme Exposure (> €1,000)"
    ]

    # Vectorized tier aggregation on Amount and Class columns only
    amt_series = df["Amount"]
    has_class = "Class" in df.columns
    class_series = df["Class"] if has_class else pd.Series(0, index=df.index)
    tier_series = pd.cut(amt_series, bins=bins, labels=labels)

    tier_stats = []
    for lbl in labels:
        tier_mask = (tier_series == lbl)
        cnt = int(tier_mask.sum())
        f_cnt = int(class_series[tier_mask].sum()) if has_class else 0
        f_rate = (f_cnt / cnt * 100) if cnt > 0 else 0.0
        vol = float(amt_series[tier_mask].sum())
        mean_amt = float(amt_series[tier_mask].mean()) if cnt > 0 else 0.0

        tier_stats.append({
            "Monetary Tier": lbl,
            "Total Volume": f"{cnt:,}",
            "Fraud Cases": f"{f_cnt:,}",
            "Fraud Incident Rate": f"{f_rate:.3f}%",
            "Total Capital": f"€{vol:,.2f}",
            "Average Amount": f"€{mean_amt:,.2f}",
        })

    st.dataframe(pd.DataFrame(tier_stats), use_container_width=True, hide_index=True)

    # --- SECTION 3: Visual Analytics (Tier Risk & PCA Differentiators) ---
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                FRAUD INCIDENCE RATE BY AMOUNT BRACKET
            </div>
        """, unsafe_allow_html=True)
        tier_plot_data = pd.DataFrame(tier_stats)
        tier_plot_data["Rate_Num"] = [float(x.replace("%", "")) for x in tier_plot_data["Fraud Incident Rate"]]

        fig_tier = go.Figure(go.Bar(
            x=tier_plot_data["Monetary Tier"],
            y=tier_plot_data["Rate_Num"],
            marker=dict(
                color=tier_plot_data["Rate_Num"],
                colorscale=[[0, "#86EFAC"], [0.5, GREEN_PRIMARY], [1.0, FRAUD_RED]]
            ),
            text=tier_plot_data["Fraud Incident Rate"],
            textposition="auto"
        ))
        layout_t = get_plotly_layout(height=300)
        layout_t["yaxis"]["title"] = "Fraud Concentration (%)"
        fig_tier.update_layout(layout_t)
        st.plotly_chart(fig_tier, use_container_width=True)

    with c2:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                TOP LATENT DISCRIMINANT (V14 vs V17)
            </div>
        """, unsafe_allow_html=True)
        if "V14" in df.columns and "V17" in df.columns and has_class:
            legit_idx = df.index[df["Class"] == 0]
            sampled_legit_idx = np.random.choice(legit_idx, size=min(800, len(legit_idx)), replace=False)
            fraud_idx = df.index[df["Class"] == 1]
            sel_idx = np.concatenate([sampled_legit_idx, fraud_idx])

            comb = df.loc[sel_idx, ["V14", "V17", "Class"]].copy()
            comb["Class_Label"] = comb["Class"].map({0: "Legitimate", 1: "Fraudulent"})

            fig_scatter = px.scatter(
                comb,
                x="V14",
                y="V17",
                color="Class_Label",
                color_discrete_map={"Legitimate": GREEN_PRIMARY, "Fraudulent": FRAUD_RED},
                opacity=0.7
            )
            layout_s = get_plotly_layout(height=300)
            layout_s["xaxis"]["title"] = "V14 (Strongest Fraud Inversion Factor)"
            layout_s["yaxis"]["title"] = "V17 (Secondary Discriminant)"
            fig_scatter.update_layout(layout_s)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- SECTION 4: Empirical Dataset Dimensions & Integrity ---
    st.markdown("""
        <div style="font-size: 13px; font-weight: 700; color: #1F2937; margin: 16px 0 10px 0; letter-spacing: 0.5px; text-transform: uppercase;">
            DATASET ARCHITECTURAL DIMENSIONS
        </div>
    """, unsafe_allow_html=True)

    dims_info = [
        {"Dimension Property": "Total Recorded Observations", "Value": f"{len(full_df):,} transactions"},
        {"Dimension Property": "Observation Time Span", "Value": "48.0 Hours (172,792 seconds continuous window)"},
        {"Dimension Property": "Input Feature Count", "Value": "30 Features (Time, Amount, V1-V28 PCA Components)"},
        {"Dimension Property": "Target Class Distribution", "Value": "284,315 Legitimate (99.828%) vs 492 Fraud (0.172%)"},
        {"Dimension Property": "Missing Elements", "Value": f"{full_df.isnull().sum().sum()} missing values"},
        {"Dimension Property": "Memory Footprint", "Value": f"~{full_df.memory_usage().sum() / (1024*1024):.1f} MB"},
    ]
    st.dataframe(pd.DataFrame(dims_info), use_container_width=True, hide_index=True)
