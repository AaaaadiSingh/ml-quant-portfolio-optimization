"""
Main Streamlit Application Entry Point
======================================
Machine Learning for Quantitative Portfolio Construction
Institutional Research Dashboard & Empirical Validation Engine

Author: Adi Aditya Singh
Supervisor: Dr. Ruchika Sehgal (USAR, GGSIPU)
"""
from __future__ import annotations

import streamlit as st
from dashboard import styles
from dashboard.pages import (
    overview,
    performance,
    holdout,
    robustness,
    weights,
    methodology,
)

# 1. Page Configuration (must be first Streamlit call)
st.set_page_config(
    page_title="Quant Portfolio Construction | Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Financial Styling
styles.apply_theme()

# 3. Sidebar Navigation & Metadata
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <h2 style="margin: 0; color: #0F172A; font-size: 1.25rem; font-weight: 700;">
                📈 ML Quant Portfolio
            </h2>
            <p style="margin: 0.2rem 0 0.8rem 0; color: #64748B; font-size: 0.8rem; font-weight: 500;">
                Uncertainty Resampling Framework
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display: flex; gap: 0.35rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <span class="badge-pill badge-green">v1.0-final</span>
            <span class="badge-pill badge-blue">43/43 Tests PASS</span>
            <span class="badge-pill badge-slate">NSE 46 Stocks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        options=[
            "📊 Executive Overview",
            "📈 Strategy Performance",
            "⚖️ Development vs. Holdout",
            "🛡️ Risk & Robustness",
            "💼 Portfolio Weights Explorer",
            "📖 Methodology & Limitations",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Author & Academic Supervision
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #475569; line-height: 1.5;">
            <strong>Author:</strong> Adi Aditya Singh<br>
            <strong>Roll No:</strong> 12419051723 (IIOT-B2)<br>
            <strong>Supervisor:</strong> Dr. Ruchika Sehgal<br>
            <em>University School of Automation and Robotics (USAR), GGSIPU</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Data Window Audit Trail
    st.markdown(
        """
        <div style="font-size: 0.775rem; color: #64748B; line-height: 1.6;">
            <strong>Data Window Audit:</strong><br>
            • <strong>Dev:</strong> 2015-01-01 → 2023-12-29 (2,222 days)<br>
            • <strong>Holdout:</strong> 2024-01-01 → 2025-06-27 (368 days)<br>
            • <strong>Universe:</strong> 46 Nifty 50 Survivors<br>
            • <strong>Execution Cost:</strong> 10 bps per turn
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4. Route to Selected Page
if selected_page == "📊 Executive Overview":
    overview.render()
elif selected_page == "📈 Strategy Performance":
    performance.render()
elif selected_page == "⚖️ Development vs. Holdout":
    holdout.render()
elif selected_page == "🛡️ Risk & Robustness":
    robustness.render()
elif selected_page == "💼 Portfolio Weights Explorer":
    weights.render()
elif selected_page == "📖 Methodology & Limitations":
    methodology.render()

# 5. Global Institutional Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.775rem; color: #94A3B8; padding: 1rem 0;">
        Machine Learning for Quantitative Portfolio Construction · 16-Week Academic Research Project (USAR, GGSIPU)<br>
        Strictly for academic evaluation and quantitative research demonstration · Not investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)
