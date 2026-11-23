"""White and Green Theme, Styling, and Plotly Configurations for FRAUDGUARD.

Author: Isbandu Rahuman R.
"""

from typing import Any, Dict
import plotly.graph_objects as go
import streamlit as st

# Color Palette Specification
BG_WHITE = "#FFFFFF"
BG_LIGHT_GREY = "#F7F9F8"
PANEL_BORDER = "#E5E7EB"
PANEL_HOVER = "#F3F4F6"

GREEN_PRIMARY = "#16A34A"
GREEN_HOVER = "#15803D"
GREEN_LIGHT_BG = "rgba(22, 163, 74, 0.08)"

FRAUD_RED = "#DC2626"
FRAUD_RED_BG = "rgba(220, 38, 38, 0.08)"
AMBER_WARNING = "#D97706"
AMBER_LIGHT_BG = "rgba(217, 119, 6, 0.08)"

TEXT_DARK = "#1F2937"
TEXT_MUTED = "#4B5563"
TEXT_SUBTLE = "#6B7280"

# Backward compatibility aliases for full compatibility
CYAN_PRIMARY = GREEN_PRIMARY
TURQUOISE = GREEN_PRIMARY
BG_DARK = BG_WHITE
PANEL_BG = BG_LIGHT_GREY
ALERT_CRIMSON = FRAUD_RED
VIOLET_ACCENT = GREEN_PRIMARY
TEXT_WHITE = TEXT_DARK


CUSTOM_CSS = f"""
<style>
    /* Main Canvas & Typography */
    .stApp {{
        background-color: {BG_WHITE};
        color: {TEXT_DARK};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    
    /* Top Header Bar */
    .top-navbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: {BG_LIGHT_GREY};
        border-bottom: 1px solid {PANEL_BORDER};
        padding: 12px 24px;
        margin: -4rem -4rem 1.5rem -4rem;
        position: sticky;
        top: 0;
        z-index: 999;
    }}
    
    .brand-title {{
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: {TEXT_DARK};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .brand-accent {{
        color: {GREEN_PRIMARY};
    }}
    
    .brand-sub {{
        font-size: 11px;
        color: {TEXT_SUBTLE};
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }}

    .user-badge {{
        display: flex;
        align-items: center;
        gap: 10px;
        background: {BG_WHITE};
        border: 1px solid {PANEL_BORDER};
        border-radius: 20px;
        padding: 6px 14px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {BG_LIGHT_GREY};
        border-right: 1px solid {PANEL_BORDER};
    }}

    /* Dashboard Panels & Cards */
    .analytics-card {{
        background: {BG_LIGHT_GREY};
        border: 1px solid {PANEL_BORDER};
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: border-color 0.2s ease;
    }}
    .analytics-card:hover {{
        border-color: {GREEN_PRIMARY};
    }}

    .kpi-title {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: {TEXT_SUBTLE};
        text-transform: uppercase;
        margin-bottom: 4px;
    }}

    .kpi-value {{
        font-size: 26px;
        font-weight: 800;
        color: {TEXT_DARK};
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}

    .kpi-meta {{
        font-size: 12px;
        margin-top: 6px;
        color: {TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Risk Badges */
    .badge-critical {{
        background-color: {FRAUD_RED_BG};
        color: {FRAUD_RED};
        border: 1px solid {FRAUD_RED};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-high {{
        background-color: {FRAUD_RED_BG};
        color: {FRAUD_RED};
        border: 1px solid {FRAUD_RED};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-medium {{
        background-color: {AMBER_LIGHT_BG};
        color: {AMBER_WARNING};
        border: 1px solid {AMBER_WARNING};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }}
    .badge-low {{
        background-color: {GREEN_LIGHT_BG};
        color: {GREEN_PRIMARY};
        border: 1px solid {GREEN_PRIMARY};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }}

    /* Section Headers */
    .section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        border-bottom: 1px solid {PANEL_BORDER};
        padding-bottom: 8px;
    }}
    .section-title {{
        font-size: 16px;
        font-weight: 700;
        color: {TEXT_DARK};
        letter-spacing: 0.3px;
    }}
    .section-tag {{
        font-size: 11px;
        color: {GREEN_PRIMARY};
        font-weight: 700;
        background: {GREEN_LIGHT_BG};
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid {GREEN_PRIMARY};
    }}

    /* Buttons */
    button[kind="primary"] {{
        background-color: {GREEN_PRIMARY} !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
    }}
    button[kind="primary"]:hover {{
        background-color: {GREEN_HOVER} !important;
    }}
    button[kind="secondary"] {{
        background-color: {BG_WHITE} !important;
        color: {TEXT_DARK} !important;
        border: 1px solid {PANEL_BORDER} !important;
        border-radius: 6px !important;
    }}
    button[kind="secondary"]:hover {{
        border-color: {GREEN_PRIMARY} !important;
        color: {GREEN_PRIMARY} !important;
    }}

    /* Dividers and borders */
    hr {{
        border-color: {PANEL_BORDER} !important;
    }}

    /* Radio navigation styling */
    div[role="radiogroup"] label {{
        color: {TEXT_DARK} !important;
    }}

    /* Custom Scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_LIGHT_GREY};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {PANEL_BORDER};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {GREEN_PRIMARY};
    }}
</style>
"""


def apply_theme():
    """Inject clean white and green CSS into current Streamlit view."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_plotly_layout(title: str = "", height: int = 320) -> Dict[str, Any]:
    """Return unified white and green styling for Plotly charts."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_DARK, size=13, weight=700)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
        margin=dict(t=35 if title else 20, b=25, l=35, r=20),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color=TEXT_MUTED, size=11),
        ),
        xaxis=dict(
            gridcolor=PANEL_BORDER,
            zerolinecolor=PANEL_BORDER,
            tickfont=dict(color=TEXT_MUTED)
        ),
        yaxis=dict(
            gridcolor=PANEL_BORDER,
            zerolinecolor=PANEL_BORDER,
            tickfont=dict(color=TEXT_MUTED)
        ),
    )
