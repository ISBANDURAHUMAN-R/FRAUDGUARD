"""FRAUDGUARD - Enterprise Financial Fraud Intelligence & Analytics Platform.

Developer & Owner: Isbandu Rahuman R.
"""

from pathlib import Path
import streamlit as st

from app.bulk_analysis import render_bulk_analysis_page
from app.fraud_monitoring import render_fraud_monitoring_page
from app.history import render_history_page
from app.model_insights import render_model_insights_page
from app.overview import render_overview_page
from app.prediction import render_prediction_page
from app.settings import render_settings_page
from app.sidebar_filters import render_sidebar_filters
from app.theme import (
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
    apply_theme,
)
from app.transaction_analysis import render_transaction_analysis_page
from src.data_loader import is_dataset_available
from src.database import get_history_stats, init_db
from src.predict import load_artifacts

# Page Configuration - Clean title without emojis
st.set_page_config(
    page_title="FRAUDGUARD | Financial Fraud Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply unified clean white & green theme
apply_theme()


def render_top_navigation():
    """Render top navigation bar with branding, quick links, and user profile."""
    # Clean navigation choices without emojis
    nav_pages = [
        "Overview",
        "Transaction Analysis",
        "Single Prediction",
        "Bulk Scanner",
        "Fraud Monitoring",
        "Transaction History",
        "Model Intelligence",
        "Settings"
    ]

    # Session state tracking for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    # Top Brand and Profile Header Bar
    head_c1, head_c2, head_c3 = st.columns([1.5, 2.5, 1.2])

    with head_c1:
        st.markdown("""
            <div style="padding: 4px 0;">
                <div style="font-size: 20px; font-weight: 800; color: #1F2937; letter-spacing: 0.5px; line-height: 1.1;">
                    FRAUD<span style="color: #16A34A;">GUARD</span>
                </div>
                <div style="font-size: 10px; color: #6B7280; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">
                    Financial Intelligence Platform
                </div>
            </div>
        """, unsafe_allow_html=True)

    with head_c2:
        chosen_page = st.radio(
            "Navigation",
            nav_pages,
            index=nav_pages.index(st.session_state.current_page) if st.session_state.current_page in nav_pages else 0,
            horizontal=True,
            label_visibility="collapsed"
        )
        if chosen_page != st.session_state.current_page:
            st.session_state.current_page = chosen_page
            st.rerun()

    with head_c3:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px; padding: 4px 0;">
                <div style="text-align: right;">
                    <div style="font-size: 13px; font-weight: 700; color: #1F2937;">Isbandu Rahuman R.</div>
                    <div style="font-size: 10px; color: #16A34A; font-weight: 600; text-transform: uppercase;">Lead Risk Officer</div>
                </div>
                <div style="background-color: #16A34A; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #FFFFFF; font-size: 13px;">
                    IR
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #E5E7EB; margin: 10px 0 20px 0;'>", unsafe_allow_html=True)
    return st.session_state.current_page


def main():
    """Main application orchestrator."""
    init_db()

    # Sidebar Header Branding
    logo_path = Path("assets") / "logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=130)

    st.sidebar.markdown("""
        <div style="margin-top: -6px; margin-bottom: 14px;">
            <div style="font-size: 16px; font-weight: 800; color: #1F2937;">
                FRAUD<span style="color: #16A34A;">GUARD</span>
            </div>
            <div style="font-size: 10px; color: #6B7280; font-weight: 500;">
                Intelligence Against Financial Fraud
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Render Collapsible Sidebar Filters
    active_filters = render_sidebar_filters(max_amount=25000.0)

    # Sidebar System Health Status
    st.sidebar.markdown("<hr style='border-color: #E5E7EB; margin: 16px 0 10px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="font-size: 11px; font-weight: 700; color: #6B7280; letter-spacing: 0.5px; text-transform: uppercase;">
            SYSTEM TELEMETRY
        </div>
    """, unsafe_allow_html=True)

    try:
        _, _, meta = load_artifacts()
        model_name = meta.get("best_model_name", "Random Forest")
        m_status = f"<span style='color: #16A34A; font-weight: 600;'>Online</span> ({model_name})"
    except Exception:
        m_status = "<span style='color: #D97706; font-weight: 600;'>Offline</span>"

    db_stats = get_history_stats()
    db_status = f"<span style='color: #16A34A; font-weight: 600;'>Connected</span> ({db_stats['total_predictions']} records)"
    ds_status = "<span style='color: #16A34A; font-weight: 600;'>Loaded</span> (284,807 rows)" if is_dataset_available() else "<span style='color: #DC2626; font-weight: 600;'>Missing</span>"

    st.sidebar.markdown(f"""
        <div style="font-size: 11px; color: #4B5563; line-height: 1.8; margin-top: 4px;">
            <div><b>Model:</b> {m_status}</div>
            <div><b>Database:</b> {db_status}</div>
            <div><b>Dataset:</b> {ds_status}</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Developer Credit
    st.sidebar.markdown("<hr style='border-color: #E5E7EB; margin: 12px 0 10px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="font-size: 11px; color: #6B7280; line-height: 1.5;">
            Architect and Developer:<br>
            <b style="color: #16A34A;">Isbandu Rahuman R.</b><br>
            Portfolio Edition © 2026
        </div>
    """, unsafe_allow_html=True)

    # Render Top Navigation Bar & Determine Current Page
    selected_page = render_top_navigation()

    # Route to matching page module with active filters
    if selected_page == "Overview":
        render_overview_page(active_filters)
    elif selected_page == "Transaction Analysis":
        render_transaction_analysis_page(active_filters)
    elif selected_page == "Single Prediction":
        render_prediction_page()
    elif selected_page == "Bulk Scanner":
        render_bulk_analysis_page()
    elif selected_page == "Fraud Monitoring":
        render_fraud_monitoring_page(active_filters)
    elif selected_page == "Transaction History":
        render_history_page(active_filters)
    elif selected_page == "Model Intelligence":
        render_model_insights_page()
    elif selected_page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
