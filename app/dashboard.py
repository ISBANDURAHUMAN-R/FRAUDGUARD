"""Dashboard Page for FRAUDGUARD (Redirects to Overview).

Author: Isbandu Rahuman R.
"""

from app.overview import render_overview_page


def render_dashboard(filters=None):
    """Render executive overview dashboard."""
    if filters is None:
        filters = {
            "time_range_hours": (0.0, 48.0),
            "amount_range": (0.0, 25000.0),
            "fraud_status": "All Transactions",
            "risk_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "prediction_source": "All Sources",
            "filters_applied": False,
        }
    render_overview_page(filters)
