"""Collapsible Sidebar Filter Panel for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st


def render_sidebar_filters(max_amount: float = 25000.0) -> Dict[str, Any]:
    """Render the collapsible filter panel in the sidebar and return filter state."""
    st.sidebar.markdown("""
        <div style="font-size: 11px; font-weight: 700; color: #16A34A; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">
            ANALYTICS FILTERS
        </div>
    """, unsafe_allow_html=True)

    # Initialize default state in session_state
    if "filter_state" not in st.session_state:
        st.session_state.filter_state = {
            "time_range_hours": (0.0, 48.0),
            "amount_range": (0.0, float(max_amount)),
            "fraud_status": "All Transactions",
            "risk_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "prediction_source": "All Sources",
            "filters_applied": False,
        }

    # Collapsible filter expander
    with st.sidebar.expander("Filter Controls", expanded=True):
        # 1. Time range
        time_range = st.slider(
            "Time Horizon (Hours 0 - 48)",
            min_value=0.0,
            max_value=48.0,
            value=st.session_state.filter_state["time_range_hours"],
            step=1.0,
            help="Filter transactions by elapsed recording time (2-day dataset window)."
        )

        # 2. Transaction Amount Range
        amt_range = st.slider(
            "Amount Window (€)",
            min_value=0.0,
            max_value=float(max_amount),
            value=st.session_state.filter_state["amount_range"],
            step=50.0,
            format="€%.0f",
            help="Filter monetary amount boundaries."
        )

        # 3. Fraud Status
        status_opts = ["All Transactions", "Legitimate Only (Class 0)", "Fraudulent Only (Class 1)"]
        curr_status = st.session_state.filter_state["fraud_status"]
        status_idx = status_opts.index(curr_status) if curr_status in status_opts else 0
        fraud_status = st.selectbox(
            "Target Fraud Classification",
            status_opts,
            index=status_idx
        )

        # 4. Risk Level
        risk_levels = st.multiselect(
            "Risk Severity Tier",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=st.session_state.filter_state["risk_levels"]
        )

        # 5. Prediction Source
        src_opts = ["All Sources", "Dataset Baseline", "Single Prediction Log", "Bulk CSV Scan"]
        curr_src = st.session_state.filter_state["prediction_source"]
        src_idx = src_opts.index(curr_src) if curr_src in src_opts else 0
        pred_source = st.selectbox(
            "Audit Source",
            src_opts,
            index=src_idx
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Apply Filters", type="primary", use_container_width=True):
                st.session_state.filter_state = {
                    "time_range_hours": time_range,
                    "amount_range": amt_range,
                    "fraud_status": fraud_status,
                    "risk_levels": risk_levels if risk_levels else ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    "prediction_source": pred_source,
                    "filters_applied": True,
                }
                st.toast("Filters updated successfully")
                st.rerun()

        with col_btn2:
            if st.button("Clear All", type="secondary", use_container_width=True):
                st.session_state.filter_state = {
                    "time_range_hours": (0.0, 48.0),
                    "amount_range": (0.0, float(max_amount)),
                    "fraud_status": "All Transactions",
                    "risk_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    "prediction_source": "All Sources",
                    "filters_applied": False,
                }
                st.toast("Filters reset to default")
                st.rerun()

    current_filters = st.session_state.filter_state

    # Visual Filter Indicator
    if current_filters["filters_applied"]:
        st.sidebar.markdown("""
            <div style="background: rgba(22, 163, 74, 0.08); border: 1px solid #16A34A; border-radius: 4px; padding: 6px 10px; margin-top: 8px; font-size: 11px; color: #16A34A; font-weight: 600;">
                Active Filter Applied: Dashboard data filtered.
            </div>
        """, unsafe_allow_html=True)

    return current_filters


def apply_dataframe_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any]
) -> pd.DataFrame:
    """Apply active filters using vectorized numpy boolean masks for high performance."""
    if df is None or df.empty or not filters:
        return df

    t_range = filters.get("time_range_hours", (0.0, 48.0))
    a_range = filters.get("amount_range", (0.0, 25000.0))
    status_sel = filters.get("fraud_status", "All Transactions")
    risk_sel = filters.get("risk_levels", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    # Fast-path: if default settings, bypass slicing completely
    is_default = (
        t_range == (0.0, 48.0)
        and a_range[0] == 0.0
        and a_range[1] >= 25000.0
        and status_sel == "All Transactions"
        and ("Risk_Level" not in df.columns or len(risk_sel) == 4)
    )
    if is_default:
        return df

    # Vectorized boolean mask
    mask = pd.Series(True, index=df.index)

    if "Time" in df.columns and t_range != (0.0, 48.0):
        min_sec = t_range[0] * 3600.0
        max_sec = t_range[1] * 3600.0
        mask &= (df["Time"] >= min_sec) & (df["Time"] <= max_sec)

    if "Amount" in df.columns and (a_range[0] > 0.0 or a_range[1] < 25000.0):
        mask &= (df["Amount"] >= a_range[0]) & (df["Amount"] <= a_range[1])

    if "Class" in df.columns and status_sel != "All Transactions":
        if status_sel == "Legitimate Only (Class 0)":
            mask &= (df["Class"] == 0)
        elif status_sel == "Fraudulent Only (Class 1)":
            mask &= (df["Class"] == 1)

    if "Risk_Level" in df.columns and risk_sel and len(risk_sel) < 4:
        mask &= df["Risk_Level"].isin(risk_sel)

    return df[mask]
