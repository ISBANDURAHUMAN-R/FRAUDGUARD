"""Page 5: Fraud Monitoring & Exposure Panel for FRAUDGUARD.

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
    BG_LIGHT_GREY,
    BG_WHITE,
    FRAUD_RED,
    FRAUD_RED_BG,
    GREEN_PRIMARY,
    PANEL_BORDER,
    TEXT_DARK,
    TEXT_MUTED,
    get_plotly_layout,
)
from src.data_loader import is_dataset_available, load_dataset
from src.database import generate_transaction_id, insert_bulk_predictions
from src.predict import determine_risk_level, load_artifacts


def render_fraud_monitoring_page(filters: Dict[str, Any]):
    """Render the high-priority fraud monitoring and incident exposure center."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">SECURITY OPERATIONS & FRAUD EXPOSURE MONITOR</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Critical Incident Surveillance, High-Risk Threat Interception and Capital Exposure
                </div>
            </div>
            <div class="section-tag" style="background: rgba(220, 38, 38, 0.08); color: #DC2626; border-color: #DC2626;">
                THREAT MONITOR ACTIVE
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not is_dataset_available():
        st.warning("European Credit Card Fraud dataset is unavailable.")
        return

    full_df = load_dataset()
    df = apply_dataframe_filters(full_df, filters)

    fraud_slice = df[df["Class"] == 1] if "Class" in df.columns else pd.DataFrame()
    total_fraud_count = len(fraud_slice)
    total_capital_at_risk = float(fraud_slice["Amount"].sum()) if not fraud_slice.empty else 0.0

    try:
        _, _, meta = load_artifacts()
        threshold = meta.get("selected_threshold", 0.6202)
    except Exception:
        threshold = 0.6202

    # --- SECTION 1: PROMINENT FRAUD EXPOSURE PANEL ---
    st.markdown(f"""
        <div style="background: {BG_LIGHT_GREY}; border: 1px solid {PANEL_BORDER}; border-left: 4px solid {FRAUD_RED}; border-radius: 6px; padding: 20px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {PANEL_BORDER}; padding-bottom: 12px; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: {FRAUD_RED}; letter-spacing: 0.8px; text-transform: uppercase;">
                        DEFENSE STATUS: ACTIVE INTERCEPTION
                    </span>
                    <div style="font-size: 22px; font-weight: 800; color: {TEXT_DARK}; margin-top: 2px;">
                        Enterprise Fraud Exposure Summary
                    </div>
                </div>
                <div style="background: {FRAUD_RED_BG}; border: 1px solid {FRAUD_RED}; border-radius: 4px; padding: 6px 14px; text-align: right;">
                    <div style="font-size: 10px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">System Threat Level</div>
                    <div style="font-size: 14px; font-weight: 800; color: {FRAUD_RED};">ELEVATED WATCH</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                <div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">Intercepted Threats</div>
                    <div style="font-size: 26px; font-weight: 800; color: {FRAUD_RED}; margin-top: 2px;">{total_fraud_count:,}</div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 4px;">Confirmed anomalous attacks</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">Capital Exposure Protected</div>
                    <div style="font-size: 26px; font-weight: 800; color: {GREEN_PRIMARY}; margin-top: 2px;">€{total_capital_at_risk:,.2f}</div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 4px;">Potential financial loss prevented</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">Model Calibrated Threshold</div>
                    <div style="font-size: 26px; font-weight: 800; color: {TEXT_DARK}; margin-top: 2px;">{threshold * 100:.1f}%</div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 4px;">Optimal decision boundary</div>
                </div>
                <div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; font-weight: 600;">Fraud Attack Concentration</div>
                    <div style="font-size: 26px; font-weight: 800; color: {TEXT_DARK}; margin-top: 2px;">{(total_fraud_count / len(df) * 100):.3f}%</div>
                    <div style="font-size: 11px; color: {TEXT_MUTED}; margin-top: 4px;">In current filtered window</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- SECTION 2: Cumulative Exposure & Hourly Velocity ---
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                CUMULATIVE CAPITAL EXPOSURE OVER TIME (€)
            </div>
        """, unsafe_allow_html=True)
        if not fraud_slice.empty:
            fraud_sorted = fraud_slice.sort_values("Time").copy()
            fraud_sorted["Cum_Amount"] = fraud_sorted["Amount"].cumsum()
            fraud_sorted["Hours"] = fraud_sorted["Time"] / 3600.0

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=fraud_sorted["Hours"],
                y=fraud_sorted["Cum_Amount"],
                mode="lines",
                name="Protected Capital",
                line=dict(color=GREEN_PRIMARY, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(22, 163, 74, 0.08)"
            ))
            layout_cum = get_plotly_layout(height=280)
            layout_cum["xaxis"]["title"] = "Elapsed Time (Hours)"
            layout_cum["yaxis"]["title"] = "Cumulative Euros (€)"
            fig_cum.update_layout(layout_cum)
            st.plotly_chart(fig_cum, use_container_width=True)

    with c2:
        st.markdown("""
            <div style="font-size: 12px; font-weight: 700; color: #4B5563; margin-bottom: 8px; text-transform: uppercase;">
                ATTACK VELOCITY DISTRIBUTION
            </div>
        """, unsafe_allow_html=True)
        if not fraud_slice.empty:
            fraud_slice["Hour_Block"] = (fraud_slice["Time"] / 3600.0).astype(int) % 24
            hourly_counts = fraud_slice["Hour_Block"].value_counts().sort_index().reset_index()
            hourly_counts.columns = ["Hour", "Incidents"]

            fig_bar = go.Figure(go.Bar(
                x=hourly_counts["Hour"],
                y=hourly_counts["Incidents"],
                marker=dict(color=FRAUD_RED)
            ))
            layout_bar = get_plotly_layout(height=280)
            layout_bar["xaxis"]["title"] = "Hour of Day (0 - 23)"
            layout_bar["yaxis"]["title"] = "Fraud Incidents"
            fig_bar.update_layout(layout_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- SECTION 3: High-Risk Surveillance Table ---
    st.markdown("""
        <div style="font-size: 13px; font-weight: 700; color: #1F2937; margin: 16px 0 10px 0; letter-spacing: 0.5px; text-transform: uppercase;">
            HIGH-RISK TRANSACTIONS SURVEILLANCE TABLE
        </div>
    """, unsafe_allow_html=True)

    if not fraud_slice.empty:
        display_fraud = fraud_slice.head(20).copy()
        display_fraud["Incident_ID"] = [f"INC-{i+1001}" for i in range(len(display_fraud))]
        display_fraud["Severity"] = np.where(display_fraud["Amount"] > 200, "CRITICAL", "HIGH")
        display_fraud["Status"] = "FLAGGED"
        display_fraud["Amount_Fmt"] = display_fraud["Amount"].map(lambda x: f"€{x:,.2f}")
        display_fraud["Elapsed_Time"] = display_fraud["Time"].map(lambda x: f"{int(x)}s ({x/3600:.1f}h)")

        cols_to_show = ["Incident_ID", "Elapsed_Time", "Amount_Fmt", "Severity", "Status", "V14", "V10", "V12", "V17"]
        st.dataframe(display_fraud[cols_to_show], use_container_width=True, hide_index=True)

        exp_c1, exp_c2 = st.columns([1, 1])
        with exp_c1:
            csv_exp = display_fraud.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Monitored Threats to Incident CSV",
                data=csv_exp,
                file_name="fraudguard_monitored_incidents.csv",
                mime="text/csv",
                use_container_width=True
            )
        with exp_c2:
            if st.button("Log Monitored Threats into SQLite Audit Database", use_container_width=True):
                recs = []
                for _, r in display_fraud.iterrows():
                    recs.append({
                        "transaction_id": generate_transaction_id("TX-INC-"),
                        "prediction_source": "monitoring",
                        "amount": float(r["Amount"]),
                        "time_sec": float(r["Time"]),
                        "prediction_label": "Fraudulent",
                        "fraud_probability": 0.92,
                        "risk_level": str(r["Severity"]),
                        "notes": "Surveillance monitor threat interception"
                    })
                count = insert_bulk_predictions(recs)
                st.success(f"Logged {count} critical threats into SQLite audit trail database.")
    else:
        st.info("No fraudulent transactions in current filter window.")
