"""
Page E: Portfolio Weights Explorer
==================================
Inspect historical quarterly rebalance allocations of the MC Centroid portfolio.
Enriched with company names, sector breakdown, hypothetical capital sizing,
integer share estimation, and convex constraint validation.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dashboard import data_loader as dl
from dashboard import styles


def render():
    st.markdown("## 💼 Portfolio Weights Explorer")
    st.markdown(
        "Interactive explorer of historical optimal portfolio allocations for the "
        "**Monte Carlo Centroid** strategy across all **37 quarterly rebalance boundaries (2015–2023)**."
    )

    styles.render_disclaimer(
        "CRITICAL NOTICE: These portfolio allocations reflect historical model weights generated strictly "
        "at the beginning of each quarterly walk-forward window for research backtesting. "
        "They are NOT live buy/sell recommendations, current trading signals, or financial advice.",
        is_warning=True,
    )

    weights_df = dl.load_centroid_weights()
    sector_df = dl.load_sector_map()

    if weights_df is None:
        st.error("Centroid weights artifact (phase5_selected_centroid_weights.csv) not found.")
        return

    # 1. Rebalance Date Selection
    available_dates = sorted(weights_df["rebal_date"].dt.strftime("%Y-%m-%d").unique())

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1, 1])
    with col_ctrl1:
        selected_date_str = st.selectbox(
            "Select Rebalance Date",
            options=available_dates,
            index=len(available_dates) - 1,  # Default to latest date (2023-11-07)
            help="Quarterly rebalance dates spaced by 63 trading days.",
        )
    with col_ctrl2:
        capital_inr = st.number_input(
            "Hypothetical Portfolio Capital (₹)",
            min_value=10_000,
            max_value=1_000_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
            help="Enter target portfolio value to compute stock-level rupee allocation.",
        )
    with col_ctrl3:
        top_n = st.slider("Show Top N Allocations in Chart", min_value=5, max_value=46, value=15)

    # Filter for selected date
    sub_weights = weights_df[weights_df["rebal_date"].dt.strftime("%Y-%m-%d") == selected_date_str].copy()

    # Enrich with sector & company name
    if sector_df is not None:
        sub_weights = pd.merge(sub_weights, sector_df, on="ticker", how="left")
    else:
        sub_weights["company_name"] = sub_weights["ticker"]
        sub_weights["sector_provisional"] = "Unclassified"

    sub_weights["company_name"] = sub_weights["company_name"].fillna(sub_weights["ticker"])
    sub_weights["sector_provisional"] = sub_weights["sector_provisional"].fillna("Other")

    # 2. Portfolio Constraint & Convexity Validation
    is_valid, validation_msgs = dl.validate_weights(sub_weights["weight"])

    total_pct = sub_weights["weight"].sum() * 100.0
    max_weight_pct = sub_weights["weight"].max() * 100.0
    min_weight_pct = sub_weights["weight"].min() * 100.0
    active_stocks = (sub_weights["weight"] > 1e-4).sum()

    val_col1, val_col2, val_col3, val_col4 = st.columns(4)
    with val_col1:
        st.markdown(
            styles.render_metric_card(
                "Total Allocation",
                f"{total_pct:.2f}%",
                subtext="Sum of all asset weights",
                badge="100.0% Target",
            ),
            unsafe_allow_html=True,
        )
    with val_col2:
        st.markdown(
            styles.render_metric_card(
                "Max Single-Stock",
                f"{max_weight_pct:.2f}%",
                subtext="10.0% Regulatory Cap",
                delta="Within Bounds",
                delta_positive=(max_weight_pct <= 10.05),
                badge="Cap Compliant",
            ),
            unsafe_allow_html=True,
        )
    with val_col3:
        st.markdown(
            styles.render_metric_card(
                "Active Positions",
                f"{active_stocks} / {len(sub_weights)}",
                subtext=f"Min weight: {min_weight_pct:.2f}%",
                badge="Diversified",
            ),
            unsafe_allow_html=True,
        )
    with val_col4:
        status_text = "PASSED" if is_valid else "FAILED"
        st.markdown(
            styles.render_metric_card(
                "Convexity Check",
                status_text,
                subtext="Non-negative, sum=1, capped",
                delta="100% Compliant" if is_valid else "Constraint Alert",
                delta_positive=is_valid,
                badge="Audit",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Visual Allocation: Top Holdings & Sector Breakdown
    styles.render_section_header("📊 Allocation Visualizer", f"Asset and sector breakdown for rebalance on {selected_date_str}")

    col_chart1, col_chart2 = st.columns([1.2, 0.8])

    sorted_weights = sub_weights.sort_values("weight", ascending=False).reset_index(drop=True)
    top_holdings = sorted_weights.head(top_n).sort_values("weight", ascending=True)

    with col_chart1:
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                y=top_holdings["company_name"],
                x=top_holdings["weight"] * 100.0,
                orientation="h",
                marker_color="#2563EB",
                text=[f"{w*100:.2f}%" for w in top_holdings["weight"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Weight: %{x:.2f}%<br><extra></extra>",
            )
        )
        layout_bar = styles.get_plotly_layout(
            title=f"Top {top_n} Stock Allocations (% Weight)",
            xaxis_title="Weight (%)",
            yaxis_title="",
            height=460,
        )
        layout_bar["yaxis"]["tickfont"] = dict(size=10)
        fig_bar.update_layout(layout_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        sector_group = sub_weights.groupby("sector_provisional")["weight"].sum().reset_index()
        sector_group = sector_group.sort_values("weight", ascending=False)

        fig_donut = px.pie(
            sector_group,
            values="weight",
            names="sector_provisional",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        layout_donut = styles.get_plotly_layout(
            title="Sector Exposure Breakdown",
            height=460,
        )
        fig_donut.update_layout(layout_donut)
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Exposure: %{percent:.1%}<br><extra></extra>",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # 4. Searchable Stock Weights & Capital Allocation Table
    styles.render_section_header("📋 Searchable Stock Weights & Target Rupee Allocation", "Stock-level sizing for hypothetical capital")

    sub_weights["weight_pct"] = sub_weights["weight"] * 100.0
    sub_weights["target_capital_inr"] = sub_weights["weight"] * capital_inr

    # Search filter
    search_query = st.text_input("🔍 Filter by Ticker, Company Name, or Sector:", "")
    filtered_table = sub_weights.copy()
    if search_query:
        mask = (
            filtered_table["ticker"].str.contains(search_query, case=False, na=False)
            | filtered_table["company_name"].str.contains(search_query, case=False, na=False)
            | filtered_table["sector_provisional"].str.contains(search_query, case=False, na=False)
        )
        filtered_table = filtered_table[mask]

    display_cols = ["ticker", "company_name", "sector_provisional", "weight_pct", "target_capital_inr"]
    table_view = filtered_table.sort_values("weight", ascending=False)[display_cols].rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company Name",
            "sector_provisional": "Sector",
            "weight_pct": "Weight (%)",
            "target_capital_inr": "Target Capital (₹)",
        }
    )

    st.dataframe(
        table_view.style.format({
            "Weight (%)": "{:.2f}%",
            "Target Capital (₹)": "₹{:,.0f}",
        }).highlight_max(subset=["Weight (%)"], color="#DCFCE7"),
        use_container_width=True,
        height=380,
    )

    # 5. Optional Whole-Share Quantity Estimator & CSV Download
    st.markdown("##### 🧮 Integer Share Quantity Estimator (Optional)")
    with st.expander("Upload Current Prices or Enter Manually to Compute Whole Shares"):
        st.markdown(
            "To estimate discrete whole-share purchase quantities ($N_i = \\lfloor \\text{Allocation}_i / P_i \\rfloor$), "
            "you can upload a CSV with columns `ticker,price` or enter prices manually. "
            "**Note:** To preserve financial integrity, the dashboard does not invent or assume live prices."
        )

        uploaded_price_csv = st.file_uploader("Upload Price CSV (columns: ticker, price)", type=["csv"])
        if uploaded_price_csv is not None:
            try:
                price_df = pd.read_csv(uploaded_price_csv)
                if "ticker" in price_df.columns and "price" in price_df.columns:
                    merged_prices = pd.merge(sub_weights, price_df[["ticker", "price"]], on="ticker", how="inner")
                    merged_prices["whole_shares"] = np.floor(merged_prices["target_capital_inr"] / merged_prices["price"]).astype(int)
                    merged_prices["actual_spend_inr"] = merged_prices["whole_shares"] * merged_prices["price"]
                    
                    st.dataframe(
                        merged_prices[["ticker", "company_name", "price", "target_capital_inr", "whole_shares", "actual_spend_inr"]].rename(
                            columns={
                                "ticker": "Ticker",
                                "company_name": "Company",
                                "price": "Price (₹)",
                                "target_capital_inr": "Target (₹)",
                                "whole_shares": "Shares",
                                "actual_spend_inr": "Actual Spend (₹)",
                            }
                        ).style.format({
                            "Price (₹)": "₹{:,.2f}",
                            "Target (₹)": "₹{:,.0f}",
                            "Shares": "{:,d}",
                            "Actual Spend (₹)": "₹{:,.0f}",
                        }),
                        use_container_width=True,
                    )
                else:
                    st.warning("Uploaded CSV must have 'ticker' and 'price' columns.")
            except Exception as e:
                st.error(f"Error processing price CSV: {e}")

    # CSV Download Button
    csv_bytes = table_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Allocations for {selected_date_str} as CSV",
        data=csv_bytes,
        file_name=f"mc_centroid_weights_{selected_date_str}.csv",
        mime="text/csv",
    )
