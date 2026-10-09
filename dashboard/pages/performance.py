"""
Page B: Strategy Performance
============================
Interactive performance analysis: ₹1,00,000 growth curves, rolling drawdowns,
risk-adjusted metrics, and sortable comparison tables across all 8 walk-forward strategies.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard import data_loader as dl
from dashboard import styles


def render():
    st.markdown("## 📈 Walk-Forward Strategy Performance (2015–2023)")
    st.markdown(
        "Interactive analysis of 9-year out-of-sample performance across **2,222 trading days** "
        "under continuous quarterly rebalancing and 10 bps one-way transaction cost drag."
    )

    styles.render_disclaimer(
        "Equity curves reflect continuous walk-forward compounding (E_t = E_{t-1} * (1 + r_t)). "
        "At each quarterly boundary (63 trading days), models are retrained using only prior data, "
        "and turnover costs are deducted immediately upon rebalance."
    )

    eq_df = dl.load_equity_curves()
    metrics_df = dl.load_metrics_summary()

    if eq_df is None or metrics_df is None:
        st.error("Required performance artifacts (phase6_all_equities_txadj.csv or metrics_summary.csv) not found.")
        return

    # 1. Interactive Capital Growth Chart
    styles.render_section_header("💰 Cumulative Growth of Capital", "Simulated growth with daily price-drift and transaction cost friction")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1.2, 0.8])
    with ctrl_col1:
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=10_000,
            max_value=100_000_000,
            value=100_000,
            step=25_000,
            format="%d",
        )
    with ctrl_col2:
        available_strats = list(eq_df.columns)
        default_strats = ["centroid", "equal_weight", "classic_max_sharpe", "rf", "ridge_linreg"]
        default_selected = [s for s in default_strats if s in available_strats]
        selected_strats = st.multiselect(
            "Select Strategies to Compare",
            options=available_strats,
            default=default_selected,
            format_func=lambda x: dl.STRATEGY_DISPLAY.get(x, x),
        )
    with ctrl_col3:
        use_log_scale = st.checkbox("Logarithmic Scale", value=False)
        show_covid_shade = st.checkbox("Highlight COVID-19 Crash", value=True)

    if not selected_strats:
        st.warning("Please select at least one strategy to display.")
        return

    # Build Growth Figure
    fig_growth = go.Figure()
    for strat in selected_strats:
        scaled_equity = eq_df[strat] * initial_capital
        color = dl.STRATEGY_COLORS.get(strat, "#64748B")
        is_centroid = (strat == "centroid")
        lw = 3.0 if is_centroid else 1.75
        dash = "solid" if is_centroid or strat == "equal_weight" else "dot" if "point" in strat else "solid"

        fig_growth.add_trace(
            go.Scatter(
                x=eq_df.index,
                y=scaled_equity,
                mode="lines",
                name=dl.STRATEGY_DISPLAY.get(strat, strat),
                line=dict(color=color, width=lw, dash=dash),
                hovertemplate=f"<b>{dl.STRATEGY_DISPLAY.get(strat, strat)}</b><br>"
                              + "Date: %{x|%Y-%m-%d}<br>"
                              + "Portfolio Value: ₹%{y:,.0f}<br>"
                              + "<extra></extra>",
            )
        )

    # Optional COVID-19 crash shading (Feb 2020 - Apr 2020)
    if show_covid_shade:
        fig_growth.add_vrect(
            x0="2020-02-15",
            x1="2020-04-15",
            fillcolor="#FEE2E2",
            opacity=0.45,
            layer="below",
            line_width=0,
            annotation_text="COVID-19 Crash",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#991B1B"),
        )

    layout_growth = styles.get_plotly_layout(
        title=f"Cumulative Growth of ₹{initial_capital:,.0f} Investment (2015–2023)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        height=480,
    )
    if use_log_scale:
        layout_growth["yaxis"]["type"] = "log"

    fig_growth.update_layout(layout_growth)
    st.plotly_chart(fig_growth, use_container_width=True)

    # Growth Summary Cards
    st.markdown("##### 📌 Final Portfolio Value & Multiplier (at 2023-12-29)")
    card_cols = st.columns(min(len(selected_strats), 4))
    for i, strat in enumerate(selected_strats[:4]):
        final_val = float(eq_df[strat].iloc[-1] * initial_capital)
        mult = float(eq_df[strat].iloc[-1])
        cagr_val = metrics_df.loc[metrics_df["strategy"] == strat, "ann_return_pct"].values[0]
        with card_cols[i % 4]:
            st.markdown(
                styles.render_metric_card(
                    dl.STRATEGY_DISPLAY.get(strat, strat).split(" (")[0],
                    f"₹{final_val:,.0f}",
                    subtext=f"{mult:.2f}x initial capital · CAGR: {cagr_val:.1f}%",
                    badge="Selected",
                ),
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 2. Rolling Drawdown Overlay Chart
    styles.render_section_header("🌊 Rolling Drawdown Profile", "Peak-to-trough decline analysis across market cycles")

    fig_dd = go.Figure()
    for strat in selected_strats:
        eq = eq_df[strat]
        running_max = eq.cummax()
        drawdown_pct = (eq - running_max) / running_max * 100.0
        color = dl.STRATEGY_COLORS.get(strat, "#64748B")
        lw = 2.5 if strat == "centroid" else 1.25

        fig_dd.add_trace(
            go.Scatter(
                x=eq.index,
                y=drawdown_pct,
                mode="lines",
                name=dl.STRATEGY_DISPLAY.get(strat, strat),
                line=dict(color=color, width=lw),
                fill="tozeroy" if strat == "centroid" else "none",
                fillcolor="rgba(37, 99, 235, 0.08)" if strat == "centroid" else None,
                hovertemplate=f"<b>{dl.STRATEGY_DISPLAY.get(strat, strat)}</b><br>"
                              + "Date: %{x|%Y-%m-%d}<br>"
                              + "Drawdown: %{y:.2f}%<br>"
                              + "<extra></extra>",
            )
        )

    layout_dd = styles.get_plotly_layout(
        title="Rolling Drawdown Overlay (All Selected Strategies)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=380,
    )
    layout_dd["yaxis"]["ticksuffix"] = "%"
    fig_dd.update_layout(layout_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # 3. Comprehensive Performance & Risk Table
    styles.render_section_header("📊 Full Risk-Adjusted Metrics Scorecard", "Institutional return, volatility, downside, and turnover statistics")

    # Metrics table options
    sort_by = st.selectbox(
        "Sort Table By",
        options=["sharpe_txadj", "ann_return_pct", "calmar", "max_dd_pct", "ann_turnover_bps", "sortino"],
        index=0,
        format_func=lambda x: {
            "sharpe_txadj": "Sharpe Ratio (Tx-Adjusted) [Desc]",
            "ann_return_pct": "Annualized Return / CAGR [Desc]",
            "calmar": "Calmar Ratio [Desc]",
            "max_dd_pct": "Maximum Drawdown [Asc]",
            "ann_turnover_bps": "Annual Turnover (bps) [Asc]",
            "sortino": "Sortino Ratio [Desc]",
        }.get(x, x),
    )

    ascending = True if sort_by in ["max_dd_pct", "ann_turnover_bps"] else False
    sorted_df = metrics_df.sort_values(by=sort_by, ascending=ascending).copy()

    table_cols = [
        "display_name", "ann_return_pct", "ann_vol", "sharpe_txadj",
        "sortino", "max_dd_pct", "calmar", "ann_turnover_bps",
        "var_95_daily_pct", "cvar_95_daily_pct"
    ]
    renamed = {
        "display_name": "Strategy",
        "ann_return_pct": "CAGR (%)",
        "ann_vol": "Ann. Vol (%)",
        "sharpe_txadj": "Sharpe (Tx-Adj)",
        "sortino": "Sortino",
        "max_dd_pct": "Max DD (%)",
        "calmar": "Calmar",
        "ann_turnover_bps": "Turnover (bps/yr)",
        "var_95_daily_pct": "Daily VaR 95% (%)",
        "cvar_95_daily_pct": "Daily CVaR 95% (%)",
    }
    view_table = sorted_df[table_cols].rename(columns=renamed)

    st.dataframe(
        view_table.style.format({
            "CAGR (%)": "{:.2f}%",
            "Ann. Vol (%)": lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%",
            "Sharpe (Tx-Adj)": "{:.4f}",
            "Sortino": "{:.4f}",
            "Max DD (%)": "{:.2f}%",
            "Calmar": "{:.4f}",
            "Turnover (bps/yr)": "{:,.1f}",
            "Daily VaR 95% (%)": "{:.2f}%",
            "Daily CVaR 95% (%)": "{:.2f}%",
        }).highlight_max(subset=["Sharpe (Tx-Adj)", "CAGR (%)", "Sortino", "Calmar"], color="#DCFCE7")
          .highlight_min(subset=["Turnover (bps/yr)"], color="#E0F2FE"),
        use_container_width=True,
    )

    # Methodological notes
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#64748B; margin-top:0.5rem; line-height:1.6;">
            <strong>Annualization Conventions & Financial Definitions:</strong><br>
            • <strong>CAGR:</strong> Compound Annual Growth Rate over T = 2,222 trading days: (E_end / E_start)^(252 / T) − 1.<br>
            • <strong>Annualized Volatility:</strong> Standard deviation of daily log returns annualized by √252.<br>
            • <strong>Sharpe (Tx-Adj):</strong> Annualized Sharpe ratio incorporating 10 bps per unit one-way portfolio turnover: (CAGR − Rf) / Vol, with Rf = 4.0% p.a.<br>
            • <strong>Daily VaR / CVaR (95%):</strong> Empirical 5th percentile of daily portfolio return distribution and expected shortfall below the 5th percentile.<br>
            • <strong>Calmar Ratio:</strong> CAGR / |Max Drawdown|.<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
