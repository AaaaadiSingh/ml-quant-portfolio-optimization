"""
Page A: Executive Overview
==========================
High-level summary of the research questions, core methodology, headline results,
and honest out-of-sample verdict.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
from dashboard import data_loader as dl
from dashboard import styles


def render():
    st.markdown("## 📊 Executive Overview")
    st.markdown(
        "**Research Project:** *Machine Learning for Quantitative Portfolio Construction: "
        "A Multi-Layer Uncertainty Resampling Framework on Indian Equities (NSE NIFTY 50)*"
    )

    styles.render_disclaimer(
        "This research dashboard presents historical empirical findings from a quantitative academic "
        "research study. All metrics are computed out-of-sample under strict walk-forward and holdout "
        "disciplines. Past performance does not guarantee future results and this is not financial advice."
    )

    # 1. Headline Metrics Row
    metrics_df = dl.load_metrics_summary()
    holdout_df = dl.load_holdout_eval()

    col1, col2, col3, col4 = st.columns(4)

    if metrics_df is not None:
        centroid_row = metrics_df[metrics_df["strategy"] == "centroid"].iloc[0]
        ew_row = metrics_df[metrics_df["strategy"] == "equal_weight"].iloc[0]

        with col1:
            st.markdown(
                styles.render_metric_card(
                    "Top Strategy (Dev)",
                    "1.1152",
                    subtext="MC Centroid (vs 0.9739 EW)",
                    delta="+14.1 bps Sharpe",
                    delta_positive=True,
                    badge="Winner",
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                styles.render_metric_card(
                    "Turnover Reduction",
                    "6,148.9 bps",
                    subtext="vs 16,758.5 bps Ridge ML",
                    delta="−63.3% Churn",
                    delta_positive=True,
                    badge="Stability",
                ),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                styles.render_metric_card(
                    "Deflated Sharpe (DSR)",
                    "0.9967",
                    subtext="p = 0.0033 (N=24 trials)",
                    delta="Stat. Significant",
                    delta_positive=True,
                    badge="Anti-P-Hacking",
                ),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                styles.render_metric_card(
                    "Overfitting Risk (PBO)",
                    "0.300",
                    subtext="CSCV S=6 slices (< 0.50 threshold)",
                    delta="Low Overfit",
                    delta_positive=True,
                    badge="Robust",
                ),
                unsafe_allow_html=True,
            )
    else:
        st.warning("Metrics summary CSV not found. Please verify results/metrics_summary.csv.")

    st.markdown("---")

    # 2. Research Problem & Solution Architecture
    styles.render_section_header("🎯 The Core Research Problem", "Why classical Markowitz optimization breaks down in practice")

    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown(
            """
            In modern quantitative portfolio management, classical **Markowitz Mean-Variance Optimization (MVO)**
            treats estimated expected returns ($\hat{\mu}$) and covariances ($\hat{\Sigma}$) as known constants.
            In reality, return estimates are notoriously noisy. As famously proved by Michaud (1989), mean-variance
            optimizers act as **'error maximizers'**:
            
            - **Over-allocation to lucky estimates:** The optimizer overweights assets whose historical sample returns were overstated due to noise.
            - **Extreme corner solutions:** Portfolios concentrate in a handful of high-beta names.
            - **Excessive turnover:** Small changes in quarterly estimates cause 100%+ portfolio reallocations, eroding alpha through execution costs.
            
            **The Proposed Multi-Layer Framework:**
            1. **ML Estimation Engine:** Pooled walk-forward regression (Ridge, Random Forest, XGBoost) using 77 engineered multi-factor features.
            2. **Monte Carlo Resampling:** Generating $K=500$ plausible forward-return scenarios using empirical fat-tailed residuals ($\nu \approx 4.30$).
            3. **Centroid Weight Aggregation:** Solving continuous convex risk-budgeted allocations across each scenario and taking the centroid $\bar{\mathbf{w}} = \frac{1}{K}\sum \mathbf{w}_k$.
            """
        )

    with col_b:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem;">
                <h4 style="margin-top:0; color:#0F172A; font-size:1rem;">🔬 Research Investigation Scope</h4>
                <ul style="font-size:0.875rem; color:#334155; line-height: 1.7; padding-left: 1.2rem;">
                    <li><strong>Market:</strong> National Stock Exchange of India (NSE)</li>
                    <li><strong>Universe:</strong> 46 NIFTY 50 survivor equities (2015–2025)</li>
                    <li><strong>Dev Window:</strong> 2015-01-01 → 2023-12-29 (2,222 trading days)</li>
                    <li><strong>Strict Holdout:</strong> 2024-01-01 → 2025-06-27 (368 trading days)</li>
                    <li><strong>Friction:</strong> 10 bps one-way turnover drag</li>
                    <li><strong>Rebalancing:</strong> Quarterly (63 business days)</li>
                    <li><strong>Constraints:</strong> 10% single-stock cap, ±3pp sector bounds</li>
                    <li><strong>Strategies:</strong> 8 walk-forward competitors</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Strategy Roster & Performance Snapshot Table
    styles.render_section_header("📋 Strategy Performance Scorecard", "Consolidated results from 9 years of walk-forward simulation")

    if metrics_df is not None:
        display_cols = [
            "display_name", "ann_return_pct", "ann_vol", "sharpe_txadj",
            "sortino", "max_dd_pct", "calmar", "ann_turnover_bps", "dsr_prob", "p7_win_rate_vs_ew"
        ]
        col_names = {
            "display_name": "Strategy",
            "ann_return_pct": "Ann Return (%)",
            "ann_vol": "Ann Vol (%)",
            "sharpe_txadj": "Sharpe (Tx-Adj)",
            "sortino": "Sortino",
            "max_dd_pct": "Max DD (%)",
            "calmar": "Calmar",
            "ann_turnover_bps": "Turnover (bps/yr)",
            "dsr_prob": "DSR Prob",
            "p7_win_rate_vs_ew": "Win Rate vs EW",
        }

        view_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]].copy()
        view_df.rename(columns=col_names, inplace=True)

        # Formatting
        st.dataframe(
            view_df.style.format({
                "Ann Return (%)": "{:.2f}%",
                "Ann Vol (%)": lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%",
                "Sharpe (Tx-Adj)": "{:.4f}",
                "Sortino": "{:.4f}",
                "Max DD (%)": "{:.2f}%",
                "Calmar": "{:.4f}",
                "Turnover (bps/yr)": "{:,.1f}",
                "DSR Prob": "{:.4f}",
                "Win Rate vs EW": lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—",
            }).highlight_max(subset=["Sharpe (Tx-Adj)", "Calmar"], color="#DCFCE7")
              .highlight_min(subset=["Turnover (bps/yr)"], color="#E0F2FE"),
            use_container_width=True,
            height=325,
        )
    else:
        st.info("No metrics summary available.")

    # 4. Honest Out-of-Sample Research Takeaway
    styles.render_section_header("⚖️ Executive Research Takeaway & Honest Holdout Finding", "Separating in-sample promise from out-of-sample reality")

    st.markdown(
        """
        > **Core Empirical Conclusion:**
        > **Did ML + MC resampling improve walk-forward performance over baselines?**
        > **Yes, during development (2015–2023):** Resampling optimal portfolios across Monte Carlo scenarios produced an **optimal risk-return tradeoff** (Sharpe **1.1152** vs 0.9739 Equal Weight and 0.8762 Classic Max Sharpe) while drastically stabilizing portfolio weights, cutting turnover by **~63%** relative to single point-estimate ML optimization. The performance survived multiple testing correction (DSR = 0.9967, $p = 0.0033$) and stationary block bootstrap ($P(\text{Centroid} > \text{EW}) = 64.00\%$).
        >
        > **Honest Negative / Mixed Finding on Holdout (2024–2025):**
        > When tested on the strictly embargoed 18-month holdout window (**2024-01-01 to 2025-06-27**), the Centroid strategy delivered a respectable **12.67% CAGR** and **0.6257 Sharpe**, but was outpaced by **Random Forest (14.85% CAGR, 0.7516 Sharpe)** and naive **Equal Weight (13.40% CAGR, 0.6713 Sharpe)**. During the strong 2024 Indian market momentum rally, un-rebalanced fixed-weight allocations in equal weighting benefited from broad market beta, whereas the Centroid's defensive sector diversification dampened upside participation. Importantly, Centroid maintained superior downside controls (Max DD -17.15% vs -21.42% Classic Max Sharpe).
        """
    )
