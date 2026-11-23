"""Page 6: Transaction History & Immutable Audit Log for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
)
from src.database import (
    clear_history,
    delete_prediction,
    get_history,
    get_history_stats,
    init_db,
)


def render_history_page(filters: Optional[Dict[str, Any]] = None):
    """Render the transaction history and audit trail."""
    init_db()

    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">IMMUTABLE TRANSACTION AUDIT TRAIL</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Persistent SQLite Record Store: Search, Filter, Inspect, and Export Audited Transactions
                </div>
            </div>
            <div class="section-tag">AUDIT LOG ONLINE</div>
        </div>
    """, unsafe_allow_html=True)

    # Top KPI summary cards
    stats = get_history_stats()
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">TOTAL AUDITED</div>
                <div class="kpi-value">{stats['total_predictions']:,}</div>
                <div class="kpi-meta">Recorded Inferences</div>
            </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">FRAUD INTERCEPTIONS</div>
                <div class="kpi-value" style="color: {FRAUD_RED};">{stats['fraud_count']:,}</div>
                <div class="kpi-meta" style="color: {FRAUD_RED};">Flagged Incidents</div>
            </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">AVERAGE RISK PROBABILITY</div>
                <div class="kpi-value" style="color: {GREEN_PRIMARY};">{stats['avg_fraud_probability'] * 100:.1f}%</div>
                <div class="kpi-meta">Portfolio Mean</div>
            </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
            <div class="analytics-card">
                <div class="kpi-title">VOLUME SCREENED</div>
                <div class="kpi-value" style="color: {GREEN_PRIMARY};">€{stats['total_volume_analyzed']:,.2f}</div>
                <div class="kpi-meta">Total Checked Capital</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Search & Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1, 1, 1])

    with f_col1:
        search_query = st.text_input("Search Transaction ID or Notes", placeholder="e.g. TX-SGL- or TX-BLK-")

    with f_col2:
        risk_filter = st.selectbox(
            "Risk Level",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )

    with f_col3:
        source_filter = st.selectbox(
            "Prediction Source",
            ["All", "Single", "Bulk", "Monitoring"]
        )

    with f_col4:
        page_size = st.selectbox(
            "Rows per Page",
            [10, 25, 50, 100],
            index=1
        )

    # Pagination state
    if "history_page" not in st.session_state:
        st.session_state.history_page = 0

    # Fetch data from SQLite
    offset = st.session_state.history_page * page_size
    records, total_matching = get_history(
        search=search_query if search_query.strip() else None,
        risk_filter=risk_filter,
        source_filter=source_filter,
        limit=page_size,
        offset=offset
    )

    total_pages = max(1, (total_matching + page_size - 1) // page_size)
    st.markdown(f"**Showing {len(records)} of {total_matching} records (Page {st.session_state.history_page + 1} of {total_pages})**")

    if not records:
        st.info("No transaction records match the selected filters. Use Single Transaction or Bulk Scanner to evaluate transactions.")
    else:
        df_display = pd.DataFrame(records)
        df_display["amount"] = df_display["amount"].map(lambda x: f"€{x:,.2f}")
        df_display["fraud_probability"] = df_display["fraud_probability"].map(lambda x: f"{x * 100:.2f}%")

        rename_map = {
            "transaction_id": "Transaction ID",
            "timestamp": "Timestamp",
            "prediction_source": "Source",
            "amount": "Amount",
            "time_sec": "Time (s)",
            "prediction_label": "Prediction",
            "fraud_probability": "Fraud Prob",
            "risk_level": "Risk Level",
            "notes": "Notes"
        }
        df_clean = df_display[[
            "transaction_id", "timestamp", "prediction_source", "amount",
            "prediction_label", "fraud_probability", "risk_level", "notes"
        ]].rename(columns=rename_map)

        st.dataframe(df_clean, use_container_width=True, hide_index=True)

        # Pagination controls
        p_prev, p_info, p_next = st.columns([1, 2, 1])
        with p_prev:
            if st.button("Previous Page", disabled=(st.session_state.history_page == 0), use_container_width=True):
                st.session_state.history_page -= 1
                st.rerun()
        with p_info:
            st.markdown(f"<div style='text-align: center; color: #4B5563; padding-top: 8px;'>Page {st.session_state.history_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with p_next:
            if st.button("Next Page", disabled=(st.session_state.history_page >= total_pages - 1), use_container_width=True):
                st.session_state.history_page += 1
                st.rerun()

    st.markdown("<hr style='border-color: #E5E7EB;'>", unsafe_allow_html=True)

    # Management Actions: Export & Purge
    st.markdown("### Audit Trail Administration")
    a_col1, a_col2 = st.columns(2)

    with a_col1:
        all_export_records, _ = get_history(limit=5000, offset=0)
        if all_export_records:
            export_df = pd.DataFrame(all_export_records)
            csv_data = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Filtered Audit Trail (CSV)",
                data=csv_data,
                file_name="fraudguard_audit_trail_export.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("Export Audit Trail (CSV)", disabled=True, use_container_width=True)

    with a_col2:
        with st.expander("Database Purge Control", expanded=False):
            st.warning("Clearing history permanently removes all transaction records from the SQLite database.")
            confirm_clear = st.checkbox("I confirm I want to purge all stored history records.")
            if st.button("Purge All History Records", type="secondary", disabled=not confirm_clear, use_container_width=True):
                count = clear_history()
                st.success(f"Purged {count} records. Database is now clean.")
                st.session_state.history_page = 0
                st.rerun()
