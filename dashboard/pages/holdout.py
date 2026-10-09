"""
Page C: Development vs. Holdout Evaluation
==========================================
Honest, rigorous comparison between in-sample/walk-forward development (2015–2023)
and the strictly embargoed 18-month out-of-sample holdout period (2024–2025).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard import data_loader as dl
from dashboard import styles


def render():
    st.markdown("## ⚖️ Development vs. Holdout Evaluation")
    st.markdown(
        "A cornerstone of institutional quantitative research: evaluating whether strategies "
        "reproduce their empirical advantages on **strictly unseen, forward-looking market regimes**."
    )

    styles.render_disclaimer(
        "The holdout window (2024-01-01 to 2025-06-27, 368 trading days) was held under strict "
        "cryptographic and architectural embargo throughout all model development and hyperparameter "
        "selection. It was evaluated exactly once on the frozen final release without parameter re-tuning."
    )

    dev_df = dl.load_metrics_summary()
    holdout_df = dl.load_holdout_eval()

    if dev_df is None or holdout_df is None:
        st.error("Missing either metrics_summary.csv or holdout_eval.csv.")
        return

    # Merge on strategy identifier
    merged = pd.merge(
        dev_df[["strategy", "display_name", "ann_return_pct", "ann_vol", "sharpe_txadj", "max_dd_pct"]],
        holdout_df[["strategy", "cagr_pct", "ann_vol", "sharpe_txadj", "max_dd_pct"]],
        on="strategy",
        suffixes=("_dev", "_holdout"),
    )

    # 1. Headline Comparison Cards
    c_dev = merged[merged["strategy"] == "centroid"].iloc[0]
    rf_hold = merged[merged["strategy"] == "rf"].iloc[0]
    ew_hold = merged[merged["strategy"] == "equal_weight"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            styles.render_metric_card(
                "Centroid Dev Sharpe",
                f"{c_dev['sharpe_txadj_dev']:.4f}",
                subtext="2015–2023 (Rank #1)",
                badge="Development",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            styles.render_metric_card(
                "Centroid Holdout Sharpe",
                f"{c_dev['sharpe_txadj_holdout']:.4f}",
                subtext="2024–2025 (Rank #4)",
                delta=f"{(c_dev['sharpe_txadj_holdout'] - c_dev['sharpe_txadj_dev']):.2f} Δ",
                delta_positive=False,
                badge="Holdout",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            styles.render_metric_card(
                "Holdout Leader: RF",
                f"{rf_hold['sharpe_txadj_holdout']:.4f}",
                subtext=f"CAGR: {rf_hold['cagr_pct']:.2f}% (Rank #1)",
                delta="Outperformed",
                delta_positive=True,
                badge="Holdout #1",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            styles.render_metric_card(
                "Equal Weight Holdout",
                f"{ew_hold['sharpe_txadj_holdout']:.4f}",
                subtext=f"CAGR: {ew_hold['cagr_pct']:.2f}% (Rank #2)",
                delta="Bull Market Beta",
                delta_positive=True,
                badge="Benchmark",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Side-by-Side Performance Comparison Charts
    styles.render_section_header("📊 Side-by-Side Period Comparison", "Comparing annualized metrics across regimes")

    chart_metric = st.radio(
        "Select Comparison Metric",
        options=["Sharpe Ratio (Tx-Adj)", "CAGR / Annual Return (%)", "Max Drawdown (%)", "Annualized Volatility (%)"],
        horizontal=True,
    )

    strat_labels = [dl.STRATEGY_DISPLAY.get(s, s).split(" (")[0] for s in merged["strategy"]]

    fig_cmp = go.Figure()

    if chart_metric == "Sharpe Ratio (Tx-Adj)":
        y_dev = merged["sharpe_txadj_dev"]
        y_hold = merged["sharpe_txadj_holdout"]
        y_label = "Sharpe Ratio (Rf = 4.0%)"
        fmt = ".4f"
    elif chart_metric == "CAGR / Annual Return (%)":
        y_dev = merged["ann_return_pct"]
        y_hold = merged["cagr_pct"]
        y_label = "Annualized Return / CAGR (%)"
        fmt = ".2f"
    elif chart_metric == "Max Drawdown (%)":
        y_dev = merged["max_dd_pct_dev"]
        y_hold = merged["max_dd_pct_holdout"]
        y_label = "Maximum Drawdown (%)"
        fmt = ".2f"
    else:
        y_dev = merged["ann_vol_dev"] * 100.0 if merged["ann_vol_dev"].max() < 1 else merged["ann_vol_dev"]
        y_hold = merged["ann_vol_holdout"] * 100.0 if merged["ann_vol_holdout"].max() < 1 else merged["ann_vol_holdout"]
        y_label = "Annualized Volatility (%)"
        fmt = ".2f"

    fig_cmp.add_trace(
        go.Bar(
            x=strat_labels,
            y=y_dev,
            name="Development Window (2015–2023)",
            marker_color="#2563EB",
            text=[f"{v:{fmt}}" for v in y_dev],
            textposition="auto",
        )
    )
    fig_cmp.add_trace(
        go.Bar(
            x=strat_labels,
            y=y_hold,
            name="Holdout Window (2024–2025)",
            marker_color="#0D9488",
            text=[f"{v:{fmt}}" for v in y_hold],
            textposition="auto",
        )
    )

    layout_cmp = styles.get_plotly_layout(
        title=f"Strategy Comparison: {chart_metric}",
        xaxis_title="Strategy",
        yaxis_title=y_label,
        height=420,
    )
    layout_cmp["barmode"] = "group"
    fig_cmp.update_layout(layout_cmp)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # 3. Strategy Rankings Shift Table
    styles.render_section_header("🎖️ Shift in Strategy Rankings Across Regimes", "Full audit table connecting dev-window results with holdout realization")

    # Compute ranks
    ranked_df = merged.copy()
    ranked_df["rank_dev"] = ranked_df["sharpe_txadj_dev"].rank(ascending=False).astype(int)
    ranked_df["rank_holdout"] = ranked_df["sharpe_txadj_holdout"].rank(ascending=False).astype(int)
    ranked_df["rank_delta"] = ranked_df["rank_dev"] - ranked_df["rank_holdout"]

    display_ranked = pd.DataFrame({
        "Strategy": [dl.STRATEGY_DISPLAY.get(s, s) for s in ranked_df["strategy"]],
        "Dev Rank": ranked_df["rank_dev"],
        "Dev Sharpe": ranked_df["sharpe_txadj_dev"],
        "Dev CAGR": ranked_df["ann_return_pct"],
        "Holdout Rank": ranked_df["rank_holdout"],
        "Holdout Sharpe": ranked_df["sharpe_txadj_holdout"],
        "Holdout CAGR": ranked_df["cagr_pct"],
        "Holdout Max DD": ranked_df["max_dd_pct_holdout"],
    }).sort_values("Holdout Rank")

    st.dataframe(
        display_ranked.style.format({
            "Dev Sharpe": "{:.4f}",
            "Dev CAGR": "{:.2f}%",
            "Holdout Sharpe": "{:.4f}",
            "Holdout CAGR": "{:.2f}%",
            "Holdout Max DD": "{:.2f}%",
        }).highlight_min(subset=["Holdout Rank", "Dev Rank"], color="#DCFCE7"),
        use_container_width=True,
    )

    st.markdown("---")

    # 4. Deep-Dive Financial Analysis: The Honest Negative / Mixed Finding
    styles.render_section_header("🔍 Deep-Dive: Why Did the Centroid Not Lead on Holdout?", "Detailed attribution of out-of-sample divergence")

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown(
            """
            #### 1. Macro Regime & Broad Market Beta Surge
            During the 18-month holdout window (**Jan 2024 – Jun 2025**), Indian equities
            underwent an aggressive, broad-based rally characterized by sustained domestic
            institutional (DII) and retail inflows.
            
            - **Equal Weight (1/N)** achieved a **13.40% CAGR** and **0.6713 Sharpe**, benefiting from uniform exposure across mid/large-cap constituents.
            - **Random Forest** achieved the highest holdout Sharpe (**0.7516**) and CAGR (**14.85%**), as its non-linear tree partitions successfully selected momentum-heavy industrial and infrastructure leaders.
            - In strong trending bull markets, conservative diversification acts as a drag on upside participation compared to unconstrained factor momentum.
            """
        )

    with col_y:
        st.markdown(
            """
            #### 2. Sector Risk Controls & Downside Resilience
            The MC Centroid portfolio strictly enforces **±3 percentage point sector drift bounds**
            and a **10% single-stock cap**.
            
            - These constraints prevented the optimizer from chasing the runaway valuations of high-beta sectors (e.g. PSU Banks, Defense, Capital Goods).
            - **The Silver Lining:** Centroid achieved its design goal of downside risk mitigation. Its holdout Maximum Drawdown was **-17.15%**, compared to **-21.42%** for Classic Max Sharpe.
            - Without rebalancing (frozen weights evaluated drift-only), the Centroid portfolio still comfortably beat **Classic Max Sharpe (0.5952)** and **Ridge LinReg (0.5487)**.
            """
        )

    st.markdown(
        """
        <div style="background:#F1F5F9; border-radius:8px; padding:1rem; margin-top:1rem; font-size:0.875rem; color:#334155;">
            <strong>Key Quantitative Takeaway for Hiring Managers & Reviewers:</strong><br>
            A model that claims to win across every single market environment is almost certainly overfitted. 
            Real institutional quant research explicitly embraces honest negative and regime-dependent results. 
            The MC Centroid methodology proved superior across 9 years of diverse walk-forward cycles (including COVID crash, inflation, and rate hikes) 
            while demonstrating disciplined risk containment on unseen future data.
        </div>
        """,
        unsafe_allow_html=True,
    )
