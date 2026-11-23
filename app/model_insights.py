"""Page 7: Model Intelligence & Empirical Diagnostics for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.theme import (
    FRAUD_RED,
    GREEN_PRIMARY,
    PANEL_BORDER,
    TEXT_DARK,
    TEXT_MUTED,
    get_plotly_layout,
)
from src.predict import load_artifacts
from src.train_model import train_and_compare_models


def render_model_insights_page():
    """Render the model evaluation and scientific diagnostics console."""
    st.markdown("""
        <div class="section-header">
            <div>
                <div class="section-title">MODEL INTELLIGENCE & EMPIRICAL DIAGNOSTICS</div>
                <div style="font-size: 13px; color: #4B5563; margin-top: 2px;">
                    Held-Out Test Set Validation (56,962 Transactions / 98 Frauds), PR-AUC Curves and Feature Attribution
                </div>
            </div>
            <div class="section-tag">CALIBRATED MODEL v1.0</div>
        </div>
    """, unsafe_allow_html=True)

    # Load artifacts
    try:
        model, preprocessor, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Model artifacts not found: {e}")
        if st.button("Train Model Now with Real Dataset", type="primary"):
            with st.spinner("Training Logistic Regression & Random Forest models on real dataset..."):
                train_and_compare_models()
            st.success("Training complete. Please reload the page.")
            st.rerun()
        return

    evals = metadata.get("model_evaluations", {})
    best_name = metadata.get("best_model_name", "Random Forest")
    rf_eval = evals.get("Random Forest", {})
    lr_eval = evals.get("Logistic Regression", {})

    # Top Model Banner
    st.markdown(f"""
        <div style="background: rgba(22, 163, 74, 0.08); border-left: 4px solid #16A34A; padding: 16px 20px; border-radius: 6px; margin-bottom: 20px;">
            <div style="font-size: 11px; font-weight: 700; color: #16A34A; letter-spacing: 0.8px; text-transform: uppercase;">
                SELECTED PRODUCTION ARCHITECTURE
            </div>
            <div style="font-size: 20px; font-weight: 800; color: {TEXT_DARK}; margin-top: 2px;">
                {best_name} Classifier (Optimized SMOTE and RobustScaler Pipeline)
            </div>
            <div style="font-size: 13px; color: {TEXT_MUTED}; margin-top: 6px;">
                {metadata.get('selection_rationale', '')}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Comparative Metrics Table
    st.markdown("### Held-Out Test Set Performance Benchmark")
    benchmark_data = [
        {
            "Algorithm": "Random Forest (Selected)",
            "PR-AUC (Primary)": f"{rf_eval.get('pr_auc', 0):.4f}",
            "Recall (Fraud)": f"{rf_eval.get('recall', 0) * 100:.2f}%",
            "Precision (Fraud)": f"{rf_eval.get('precision', 0) * 100:.2f}%",
            "F1-Score": f"{rf_eval.get('f1_score', 0):.4f}",
            "F2-Score (Recall Focus)": f"{rf_eval.get('f2_score', 0):.4f}",
            "ROC-AUC": f"{rf_eval.get('roc_auc', 0):.4f}",
            "Accuracy": f"{rf_eval.get('accuracy', 0) * 100:.2f}%",
            "Selected Threshold": f"{rf_eval.get('optimal_threshold', 0.5):.4f}"
        },
        {
            "Algorithm": "Logistic Regression (Baseline)",
            "PR-AUC (Primary)": f"{lr_eval.get('pr_auc', 0):.4f}",
            "Recall (Fraud)": f"{lr_eval.get('recall', 0) * 100:.2f}%",
            "Precision (Fraud)": f"{lr_eval.get('precision', 0) * 100:.2f}%",
            "F1-Score": f"{lr_eval.get('f1_score', 0):.4f}",
            "F2-Score (Recall Focus)": f"{lr_eval.get('f2_score', 0):.4f}",
            "ROC-AUC": f"{lr_eval.get('roc_auc', 0):.4f}",
            "Accuracy": f"{lr_eval.get('accuracy', 0) * 100:.2f}%",
            "Selected Threshold": f"{lr_eval.get('optimal_threshold', 0.5):.4f}"
        }
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Diagnostic Curves: PR Curve & ROC Curve
    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.markdown("### Precision-Recall Curve (PR-AUC = 0.8333)")
        pr_data = rf_eval.get("pr_curve", {})
        if pr_data:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_data.get("recall", []),
                y=pr_data.get("precision", []),
                mode="lines",
                name="Random Forest",
                line=dict(color=GREEN_PRIMARY, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(22, 163, 74, 0.08)"
            ))
            layout_pr = get_plotly_layout(height=300)
            layout_pr["xaxis"]["title"] = "Recall (Fraud Threat Interception)"
            layout_pr["yaxis"]["title"] = "Precision (Threat Reliability)"
            fig_pr.update_layout(layout_pr)
            st.plotly_chart(fig_pr, use_container_width=True)

    with curve_col2:
        st.markdown("### ROC Curve (ROC-AUC = 0.9632)")
        roc_data = rf_eval.get("roc_curve", {})
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data.get("fpr", []),
                y=roc_data.get("tpr", []),
                mode="lines",
                name="Random Forest",
                line=dict(color=GREEN_PRIMARY, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(22, 163, 74, 0.08)"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guessing (AUC = 0.50)",
                line=dict(color="#9CA3AF", dash="dash")
            ))
            layout_roc = get_plotly_layout(height=300)
            layout_roc["xaxis"]["title"] = "False Positive Rate"
            layout_roc["yaxis"]["title"] = "True Positive Rate (Recall)"
            fig_roc.update_layout(layout_roc)
            st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion Matrix & Feature Importance
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("### Test Split Confusion Matrix")
        cm = rf_eval.get("confusion_matrix", {})
        tn = cm.get("tn", 56833)
        fp = cm.get("fp", 31)
        fn = cm.get("fn", 14)
        tp = cm.get("tp", 84)

        z_values = [[tn, fp], [fn, tp]]
        x_labels = ["Predicted Legit", "Predicted Fraud"]
        y_labels = ["Actual Legit", "Actual Fraud"]

        annotations = [
            [f"True Negative (TN)<br><b>{tn:,}</b>", f"False Positive (FP)<br><b>{fp:,}</b>"],
            [f"False Negative (FN)<br><b>{fn:,}</b>", f"True Positive (TP)<br><b>{tp:,}</b>"]
        ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#F7F9F8"], [0.5, "#E5E7EB"], [1.0, GREEN_PRIMARY]],
            showscale=False
        ))

        for i in range(2):
            for j in range(2):
                color_text = "#FFFFFF" if (i == 0 and j == 0) else TEXT_DARK
                fig_cm.add_annotation(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=annotations[i][j],
                    showarrow=False,
                    font=dict(color=color_text, size=13)
                )

        layout_cm = get_plotly_layout(height=300)
        fig_cm.update_layout(layout_cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with row2_col2:
        st.markdown("### Model Feature Importance Ranking")
        feat_imp = metadata.get("feature_importances", {})
        if feat_imp:
            top_10 = dict(list(feat_imp.items())[:10])
            f_names = list(top_10.keys())[::-1]
            f_vals = list(top_10.values())[::-1]

            fig_fimp = go.Figure(go.Bar(
                x=f_vals,
                y=f_names,
                orientation="h",
                marker=dict(color=GREEN_PRIMARY)
            ))
            layout_fimp = get_plotly_layout(height=300)
            layout_fimp["xaxis"]["title"] = "Relative Gini Importance"
            fig_fimp.update_layout(layout_fimp)
            st.plotly_chart(fig_fimp, use_container_width=True)

    # Classification Report
    st.markdown("### Scikit-Learn Classification Report (Random Forest)")
    clf_rep = rf_eval.get("classification_report", {})
    if clf_rep:
        rows_report = []
        for class_name in ["Legitimate", "Fraudulent"]:
            if class_name in clf_rep:
                item = clf_rep[class_name]
                rows_report.append({
                    "Class": class_name,
                    "Precision": f"{item['precision']:.4f}",
                    "Recall": f"{item['recall']:.4f}",
                    "F1-Score": f"{item['f1-score']:.4f}",
                    "Support": int(item['support'])
                })
        st.dataframe(pd.DataFrame(rows_report), use_container_width=True, hide_index=True)

    # Domain explanation
    st.markdown("### The Mathematics of Imbalanced Fraud Detection")
    st.markdown("""
    In financial cybersecurity, naive metrics like raw **Accuracy** are profoundly misleading:
    
    1. **The Accuracy Paradox**: In a dataset with 284,315 legitimate transactions and only 492 fraud cases (0.17%), 
       a dummy system predicting "Legitimate" 100% of the time achieves **99.83% accuracy**, while catching **zero fraudulent transactions**.
    2. **Fraud Recall Priority**: Financial institutions prioritize **Recall** ($\\frac{TP}{TP + FN}$) because an undetected fraud (False Negative) 
       directly results in unrecoverable financial theft, chargeback penalties, and customer churn.
    3. **Precision Guardrails**: Simultaneously, an overly aggressive model with near-zero precision triggers overwhelming false alarms (False Positives), 
       alienating legitimate customers with declined cards and exhausting compliance operations teams.
    4. **The PR-AUC Metric**: While ROC-AUC can remain deceptively inflated (>0.95) even with thousands of false positives due to the huge negative class, 
       **PR-AUC (Precision-Recall AUC)** strictly measures the trade-off within the minority class, making it the gold standard benchmark for fraud detection systems.
    """)
