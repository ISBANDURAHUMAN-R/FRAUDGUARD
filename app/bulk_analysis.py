"""Page 4: Bulk CSV Transaction Scanner for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
    get_plotly_layout,
)
from src.data_loader import EXPECTED_FEATURES, validate_dataset
from src.database import generate_transaction_id, insert_bulk_predictions
from src.predict import (
    MODEL_DISCLAIMER,
    get_demo_samples,
    load_artifacts,
    predict_bulk,
)


def generate_sample_csv() -> bytes:
    """Generate a valid sample CSV containing 10 authentic transactions for testing."""
    demos = get_demo_samples()
    rows = []

    legit_base = demos["Legitimate Transaction Example"]
    fraud_base = demos["Confirmed Fraudulent Example"]

    for i in range(7):
        r = legit_base.copy()
        r["Time"] = float(i * 150)
        r["Amount"] = float(25.0 + i * 22.5)
        rows.append(r)

    for j in range(3):
        r = fraud_base.copy()
        r["Time"] = float(400.0 + j * 90)
        r["Amount"] = float(150.0 * (j + 1))
        rows.append(r)

    df_sample = pd.DataFrame(rows)
    return df_sample.to_csv(index=False).encode("utf-8")


def render_bulk_analysis_page():
    """Render the bulk transaction analysis engine."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">HIGH-THROUGHPUT BATCH CSV SCREENING</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Vectorized Inference, Portfolio Risk Categorization and Annotated Report Generation
                </div>
            </div>
            <div class="section-tag">BATCH INFERENCE</div>
        </div>
    """, unsafe_allow_html=True)

    # Load model artifacts
    try:
        model, preprocessor, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Production model is unavailable: {e}. Please train model from Settings.")
        return

    # Sample template download
    st.markdown("#### Verified Test Batch Template")
    t1, t2 = st.columns([1.2, 2])
    with t1:
        st.download_button(
            label="Download 10-Transaction Test CSV",
            data=generate_sample_csv(),
            file_name="fraudguard_test_sample.csv",
            mime="text/csv",
            use_container_width=True
        )
    with t2:
        st.caption("Use this pre-formatted test dataset with confirmed fraud and legitimate vectors to test batch scoring immediately.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Financial Transactions CSV Batch",
        type=["csv"],
        help="Upload CSV containing 'Time', 'V1' through 'V28', and 'Amount'."
    )

    if uploaded_file is None:
        st.info("Select or drag-and-drop a transaction CSV file above to start batch screening.")
        return

    # Load & Validate CSV
    try:
        df_uploaded = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
        return

    is_valid, validation_errors = validate_dataset(df_uploaded, require_target=False)
    if not is_valid:
        st.error("CSV Validation Failed:")
        for err in validation_errors:
            st.markdown(f"- {err}")
        return

    st.success(f"Schema verified. Ready to process {len(df_uploaded):,} transactions.")

    # Threshold customization
    threshold_default = float(metadata.get("selected_threshold", 0.6202))
    with st.expander("Decision Threshold Tuning", expanded=False):
        custom_threshold = st.slider(
            "Fraud Decision Boundary",
            min_value=0.10,
            max_value=0.95,
            value=threshold_default,
            step=0.01,
            help="Transactions with probability >= threshold will be flagged as Fraudulent."
        )

    # Process Button
    if st.button("Analyze Batch Transactions", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing batch scoring engine...")

        with st.spinner("Executing vectorized inference and risk categorization..."):
            progress_bar.progress(40, text="Scaling features with RobustScaler...")
            df_scored, summary = predict_bulk(
                df=df_uploaded,
                model=model,
                preprocessor=preprocessor,
                threshold=custom_threshold
            )
            progress_bar.progress(100, text="Batch scoring completed.")

        st.session_state["bulk_scored_df"] = df_scored
        st.session_state["bulk_summary"] = summary
        progress_bar.empty()

    if "bulk_scored_df" in st.session_state:
        df_scored = st.session_state["bulk_scored_df"]
        summary = st.session_state["bulk_summary"]

        st.markdown("<hr style='border-color: #E5E7EB;'>", unsafe_allow_html=True)
        st.markdown("### Executive Batch Surveillance Summary")

        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
                <div class="analytics-card">
                    <div class="kpi-title">TOTAL ANALYZED</div>
                    <div class="kpi-value">{summary['total_analyzed']:,}</div>
                    <div class="kpi-meta">Transactions Scored</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
                <div class="analytics-card">
                    <div class="kpi-title">SUSPICIOUS CASES</div>
                    <div class="kpi-value" style="color: {FRAUD_RED};">{summary['fraud_count']:,}</div>
                    <div class="kpi-meta" style="color: {FRAUD_RED};">Flagged for Triage</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
                <div class="analytics-card">
                    <div class="kpi-title">FRAUD DETECTION RATE</div>
                    <div class="kpi-value" style="color: {GREEN_PRIMARY};">{summary['fraud_rate_pct']:.2f}%</div>
                    <div class="kpi-meta">Batch Threat Rate</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
                <div class="analytics-card">
                    <div class="kpi-title">AVG PROBABILITY</div>
                    <div class="kpi-value" style="color: {GREEN_PRIMARY};">{summary['avg_fraud_probability'] * 100:.2f}%</div>
                    <div class="kpi-meta">Mean Batch Risk</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Charts: Risk breakdown and probability histogram
        c_col1, c_col2 = st.columns([1, 1])

        with c_col1:
            st.markdown("#### Risk Severity Breakdown")
            risk_counts = df_scored["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]
            color_map = {
                "LOW": GREEN_PRIMARY,
                "MEDIUM": "#D97706",
                "HIGH": FRAUD_RED,
                "CRITICAL": "#991B1B"
            }
            fig_risk = px.bar(
                risk_counts,
                x="Risk_Level",
                y="Count",
                color="Risk_Level",
                color_discrete_map=color_map,
                text="Count"
            )
            layout_r = get_plotly_layout(height=280)
            layout_r["showlegend"] = False
            fig_risk.update_layout(layout_r)
            st.plotly_chart(fig_risk, use_container_width=True)

        with c_col2:
            st.markdown("#### Fraud Probability Distribution")
            fig_hist = px.histogram(
                df_scored,
                x="Fraud_Probability",
                nbins=25,
                color="Prediction",
                color_discrete_map={"Legitimate": GREEN_PRIMARY, "Fraudulent": FRAUD_RED}
            )
            layout_h = get_plotly_layout(height=280)
            fig_hist.update_layout(layout_h)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Filter & View Suspicious Records
        st.markdown("### Screened Batch Transactions Table")
        filter_opt = st.radio(
            "Filter Records:",
            ["All Records", "Suspicious / High Risk Only (Fraud Flagged)", "Critical Severity Only"],
            horizontal=True
        )

        if filter_opt == "Suspicious / High Risk Only (Fraud Flagged)":
            display_view = df_scored[df_scored["Prediction"] == "Fraudulent"]
        elif filter_opt == "Critical Severity Only":
            display_view = df_scored[df_scored["Risk_Level"] == "CRITICAL"]
        else:
            display_view = df_scored

        st.dataframe(
            display_view[["Time", "Amount", "Prediction", "Fraud_Probability", "Risk_Level", "V14", "V17", "V12"]],
            use_container_width=True,
            hide_index=True
        )

        # Action Buttons: Export & Save to DB
        st.markdown("### Export & Database Operations")
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            csv_export = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Scored CSV Report",
                data=csv_export,
                file_name="fraudguard_batch_prediction_report.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with act_col2:
            if st.button("Save High-Risk Cases to SQLite Audit Trail", use_container_width=True):
                high_risk_cases = df_scored[df_scored["Prediction"] == "Fraudulent"]
                if len(high_risk_cases) == 0:
                    st.info("No fraudulent cases in batch to log.")
                else:
                    records_to_insert = []
                    for _, row in high_risk_cases.iterrows():
                        records_to_insert.append({
                            "transaction_id": generate_transaction_id("TX-BLK-"),
                            "prediction_source": "bulk",
                            "amount": float(row["Amount"]),
                            "time_sec": float(row["Time"]),
                            "prediction_label": str(row["Prediction"]),
                            "fraud_probability": float(row["Fraud_Probability"]),
                            "risk_level": str(row["Risk_Level"]),
                            "notes": "Bulk scan auto-flagged"
                        })
                    inserted_count = insert_bulk_predictions(records_to_insert)
                    st.success(f"Successfully saved {inserted_count} suspicious records to SQLite audit history.")

        st.info(f"{MODEL_DISCLAIMER}")
