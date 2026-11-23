"""Page 3: Single Transaction Inference Console for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Dict
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    FRAUD_RED,
    FRAUD_RED_BG,
    GREEN_LIGHT_BG,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
    get_plotly_layout,
)
from src.database import generate_transaction_id, insert_prediction
from src.predict import (
    MODEL_DISCLAIMER,
    get_demo_samples,
    load_artifacts,
    predict_single,
)


def render_prediction_page():
    """Render the enterprise single transaction prediction console."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">SINGLE TRANSACTION RISK SCORING ENGINE</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Calibrated Latent Feature Evaluation, Threat Probability and Anomaly Detection
                </div>
            </div>
            <div class="section-tag">REAL-TIME INFERENCE</div>
        </div>
    """, unsafe_allow_html=True)

    # Load model artifacts
    try:
        model, preprocessor, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Production model is unavailable: {e}. Please train model from Settings.")
        return

    demos = get_demo_samples()

    # Preset Quick Loaders
    st.markdown("#### Verified Test Vector Presets")
    p1, p2, p3 = st.columns([1.2, 1.2, 2])
    with p1:
        if st.button("Load Verified Legitimate Case", use_container_width=True):
            st.session_state.feature_state = demos["Legitimate Transaction Example"].copy()
            st.toast("Loaded authentic legitimate transaction vector")
    with p2:
        if st.button("Load Confirmed Fraud Case", use_container_width=True):
            st.session_state.feature_state = demos["Confirmed Fraudulent Example"].copy()
            st.toast("Loaded authentic fraudulent transaction vector")
    with p3:
        st.caption("Load authentic test vectors from the European cardholder dataset to observe calibrated risk discrimination.")

    if "feature_state" not in st.session_state:
        st.session_state.feature_state = demos["Legitimate Transaction Example"].copy()

    curr_vals = st.session_state.feature_state

    # Primary Inputs
    st.markdown("#### Monetary and Temporal Attributes")
    c_amt, c_time = st.columns(2)

    with c_amt:
        amount_val = st.number_input(
            "Transaction Amount (€)",
            min_value=0.0,
            max_value=100000.0,
            value=float(curr_vals.get("Amount", 0.0)),
            step=10.0,
            format="%.2f",
            help="Monetary transaction amount in Euros."
        )

    with c_time:
        time_val = st.number_input(
            "Time Offset (Elapsed Seconds in 48-Hour Window)",
            min_value=0.0,
            max_value=200000.0,
            value=float(curr_vals.get("Time", 0.0)),
            step=60.0,
            format="%.1f",
            help="Seconds elapsed since start of dataset recording window."
        )

    # PCA Components Grid
    st.markdown("#### Confidential Latent Dimensions (V1 to V28)")
    pca_t1, pca_t2 = st.tabs(["Latent Components V1 - V14", "Latent Components V15 - V28"])
    features_input: Dict[str, float] = {"Amount": amount_val, "Time": time_val}

    with pca_t1:
        cols_1 = st.columns(4)
        for i in range(1, 15):
            k = f"V{i}"
            with cols_1[(i - 1) % 4]:
                features_input[k] = st.number_input(
                    k,
                    value=float(curr_vals.get(k, 0.0)),
                    format="%.5f",
                    key=f"feat_{k}"
                )

    with pca_t2:
        cols_2 = st.columns(4)
        for j in range(15, 29):
            k = f"V{j}"
            with cols_2[(j - 15) % 4]:
                features_input[k] = st.number_input(
                    k,
                    value=float(curr_vals.get(k, 0.0)),
                    format="%.5f",
                    key=f"feat_{k}"
                )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Inference Execution
    if st.button("Execute Fraud Analysis", type="primary", use_container_width=True):
        with st.spinner("Scoring telemetry vector against production Random Forest..."):
            try:
                res = predict_single(
                    features_input=features_input,
                    model=model,
                    preprocessor=preprocessor,
                    threshold=metadata.get("selected_threshold", 0.6202)
                )
            except Exception as e:
                st.error(f"Inference failure: {e}")
                return

        st.markdown("<hr style='border-color: #E5E7EB;'>", unsafe_allow_html=True)

        # Result Banner
        if res["is_fraud"]:
            st.markdown(f"""
                <div style="background: {FRAUD_RED_BG}; border: 1px solid {FRAUD_RED}; border-radius: 6px; padding: 16px 20px; margin-bottom: 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: {FRAUD_RED}; letter-spacing: 0.8px;">INTERCEPTION ADVISORY</div>
                            <div style="font-size: 20px; font-weight: 800; color: {FRAUD_RED}; margin-top: 2px;">
                                POTENTIALLY FRAUDULENT TRANSACTION DETECTED
                            </div>
                        </div>
                        <span class="badge-critical" style="font-size: 13px; padding: 5px 12px;">SEVERITY: {res['risk_level']}</span>
                    </div>
                    <div style="font-size: 13px; color: {TEXT_MUTED}; margin-top: 6px;">
                        Estimated Fraud Probability: <b>{res['fraud_probability_pct']}%</b> (Production Threshold: {res['threshold_used']*100:.1f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: {GREEN_LIGHT_BG}; border: 1px solid {GREEN_PRIMARY}; border-radius: 6px; padding: 16px 20px; margin-bottom: 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: {GREEN_PRIMARY}; letter-spacing: 0.8px;">CLEARANCE ADVISORY</div>
                            <div style="font-size: 20px; font-weight: 800; color: {GREEN_PRIMARY}; margin-top: 2px;">
                                TRANSACTION CLEARED AS LEGITIMATE
                            </div>
                        </div>
                        <span class="badge-low" style="font-size: 13px; padding: 5px 12px;">SEVERITY: {res['risk_level']}</span>
                    </div>
                    <div style="font-size: 13px; color: {TEXT_MUTED}; margin-top: 6px;">
                        Estimated Fraud Probability: <b>{res['fraud_probability_pct']}%</b> (Well below threshold: {res['threshold_used']*100:.1f}%)
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Gauge & Diagnostic Breakdown
        g_col1, g_col2 = st.columns([1, 1.1])

        with g_col1:
            st.markdown("#### Threat Probability Gauge")
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["fraud_probability_pct"],
                number=dict(suffix="%"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=TEXT_MUTED),
                    bar=dict(color=GREEN_PRIMARY if not res["is_fraud"] else FRAUD_RED),
                    steps=[
                        {"range": [0, 20], "color": "rgba(22, 163, 74, 0.12)"},
                        {"range": [20, 62], "color": "rgba(217, 119, 6, 0.12)"},
                        {"range": [62, 85], "color": "rgba(220, 38, 38, 0.15)"},
                        {"range": [85, 100], "color": "rgba(220, 38, 38, 0.3)"},
                    ],
                    threshold=dict(
                        line=dict(color=TEXT_DARK, width=2.5),
                        thickness=0.75,
                        value=res["threshold_used"] * 100
                    )
                )
            ))
            g_layout = get_plotly_layout(height=260)
            gauge_fig.update_layout(g_layout)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with g_col2:
            st.markdown("#### Assessment Diagnostics")
            st.markdown(f"""
                <div class="analytics-card" style="margin-bottom: 8px;">
                    <div class="kpi-title">TRANSACTION VALUE</div>
                    <div style="font-size: 20px; font-weight: 800; color: {TEXT_DARK};">€{res['amount']:,.2f}</div>
                </div>
                <div class="analytics-card" style="margin-bottom: 8px;">
                    <div class="kpi-title">ANOMALOUS PCA DIMENSIONS (|z| >= 3)</div>
                    <div style="font-size: 13px; color: {GREEN_PRIMARY}; font-weight: 600; margin-top: 4px;">
                        {', '.join(res['notable_features']) if res['notable_features'] else 'None (Nominal financial behavior)'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Audit Logging
        st.markdown("#### Audit Trail Recording")
        l_c1, l_c2 = st.columns([2, 1])
        with l_c1:
            notes = st.text_input("Reviewer Compliance Notes", placeholder="e.g. Confirmed with customer via phone")
        with l_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Save Record to SQLite", use_container_width=True):
                tx_id = generate_transaction_id("TX-SGL-")
                insert_prediction(
                    transaction_id=tx_id,
                    prediction_source="single",
                    amount=res["amount"],
                    time_sec=res["time_sec"],
                    prediction_label=res["prediction_label"],
                    fraud_probability=res["fraud_probability"],
                    risk_level=res["risk_level"],
                    notes=notes
                )
                st.success(f"Transaction successfully persisted to audit trail: {tx_id}")

        st.info(f"{MODEL_DISCLAIMER}")
