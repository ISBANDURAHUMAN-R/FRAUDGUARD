"""Page 8: System Settings, Dataset Management & Diagnostics for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from pathlib import Path
import sys
import pandas as pd
import sklearn
import streamlit as st

from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    TEXT_DARK,
    TEXT_MUTED,
)
from src.data_loader import (
    DEFAULT_DATA_PATH,
    get_dataset_summary,
    is_dataset_available,
    load_dataset,
    validate_dataset,
)
from src.database import clear_history, get_history, get_history_stats, init_db
from src.predict import load_artifacts
from src.train_model import train_and_compare_models


def render_settings_page():
    """Render settings, dataset administration, and model controls."""
    init_db()

    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">SYSTEM SETTINGS & REPOSITORY ADMINISTRATION</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Dataset Health, Model Calibration, Database Maintenance and Runtime Diagnostics
                </div>
            </div>
            <div class="section-tag">ADMIN CONSOLE</div>
        </div>
    """, unsafe_allow_html=True)

    tab_data, tab_model, tab_db, tab_env = st.tabs([
        "Dataset Repository",
        "Model Calibration & Retraining",
        "Database Maintenance",
        "System & Runtime Environment"
    ])

    # --- TAB 1: DATASET REPOSITORY ---
    with tab_data:
        st.markdown("#### European Credit Card Dataset Repository")
        if is_dataset_available():
            size_mb = DEFAULT_DATA_PATH.stat().st_size / (1024 * 1024)
            st.success(f"Active dataset: data/creditcard.csv ({size_mb:.1f} MB verified on disk).")
            df = load_dataset()
            summary = get_dataset_summary(df)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Records", f"{summary['total_records']:,}")
            with c2:
                st.metric("Fraud Cases", f"{summary['fraudulent_count']:,}")
            with c3:
                st.metric("Fraud Proportion", f"{summary['fraud_rate_pct']:.4f}%")
            with c4:
                st.metric("Input Features", f"{summary['total_features']}")
        else:
            st.error("Dataset creditcard.csv not detected in data/.")
            st.markdown("""
                **Instructions to install dataset:**
                1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
                2. Place the file in the `data/` directory or upload it using the form below:
            """)

        st.markdown("#### Upload or Replace Dataset")
        uploaded_csv = st.file_uploader(
            "Upload creditcard.csv (Enforces 30 features + Class schema)",
            type=["csv"],
            key="settings_uploader"
        )
        if uploaded_csv is not None:
            if st.button("Validate & Deploy Uploaded Dataset", type="primary"):
                try:
                    df_new = pd.read_csv(uploaded_csv)
                    is_valid, errs = validate_dataset(df_new, require_target=True)
                    if not is_valid:
                        st.error(f"Validation failed: {'; '.join(errs)}")
                    else:
                        DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                        df_new.to_csv(DEFAULT_DATA_PATH, index=False)
                        st.success(f"Successfully saved {len(df_new):,} records to data/creditcard.csv.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to process file: {e}")

    # --- TAB 2: MODEL CALIBRATION & RETRAINING ---
    with tab_model:
        st.markdown("#### Model Hyperparameters & Production Tuning")
        try:
            _, _, meta = load_artifacts()
            curr_thresh = float(meta.get("selected_threshold", 0.6202))
            st.markdown(f"""
                - **Production Algorithm:** `{meta.get('best_model_name', 'Random Forest')}`
                - **Current Calibrated Threshold:** `{curr_thresh:.4f}`
                - **Last Trained Timestamp:** `{meta.get('trained_at', 'N/A')}`
                - **Training Samples:** `{meta.get('train_samples', 0):,}`
            """)
        except Exception:
            curr_thresh = 0.6202
            st.info("No trained model loaded yet.")

        st.markdown("#### Trigger Retraining Pipeline")
        smote_ratio = st.slider("SMOTE Oversampling Ratio (Training Split)", 0.05, 0.30, 0.10, 0.01)
        r_seed = st.number_input("Reproducibility Seed (Random State)", value=42, step=1)

        if st.button("Retrain Production Models Now", type="primary"):
            if not is_dataset_available():
                st.error("Cannot retrain: data/creditcard.csv is missing.")
            else:
                with st.spinner("Training Logistic Regression and Random Forest with SMOTE resampling..."):
                    res = train_and_compare_models(random_state=int(r_seed))
                st.success(f"Retraining successful. Best selected algorithm: {res['best_model_name']}.")
                st.rerun()

    # --- TAB 3: DATABASE MAINTENANCE ---
    with tab_db:
        st.markdown("#### SQLite Transaction History Maintenance")
        db_stats = get_history_stats()
        st.markdown(f"""
            - **Total Stored Predictions:** `{db_stats['total_predictions']:,}`
            - **Logged Fraud Incidents:** `{db_stats['fraud_count']:,}`
            - **Total Financial Volume Logged:** `€{db_stats['total_volume_analyzed']:,.2f}`
        """)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            records, _ = get_history(limit=5000)
            if records:
                csv_db = pd.DataFrame(records).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Full SQLite Database (CSV)",
                    data=csv_db,
                    file_name="fraudguard_sqlite_backup.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with d_col2:
            st.markdown("##### Danger Zone: Database Purge")
            confirm_p = st.checkbox("Confirm database wipe")
            if st.button("Clear All Stored History", disabled=not confirm_p, type="secondary"):
                c = clear_history()
                st.success(f"Cleared {c} records from SQLite database.")
                st.rerun()

    # --- TAB 4: ENVIRONMENT & SYSTEM ---
    with tab_env:
        st.markdown("#### Environment & Diagnostic Manifest")
        st.markdown(f"""
            - **Platform:** `Windows x86_64`
            - **Python Runtime:** `Python {sys.version.split()[0]}`
            - **Scikit-Learn Version:** `{sklearn.__version__}`
            - **Streamlit Version:** `{st.__version__}`
            - **Architecture:** Full-Stack Data Science & Machine Learning Web Platform
            - **Developer & Owner:** `Isbandu Rahuman R.`
            - **Portfolio Edition:** `2026`
        """)
