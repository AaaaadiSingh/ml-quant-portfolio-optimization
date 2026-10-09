import pandas as pd
from pathlib import Path

root = Path.cwd()
source = root / "data" / "processed" / "phase5_selected_centroid_weights.csv"
out = root / "results" / "final_stock_weights.csv"

df = pd.read_csv(source)
df["rebal_date"] = pd.to_datetime(df["rebal_date"])
df["weight"] = pd.to_numeric(df["weight"], errors="raise")

# Select the latest rebalance date present in the file.
latest_date = df["rebal_date"].max()
latest = df.loc[df["rebal_date"] == latest_date].copy()

if latest.empty:
    raise ValueError("No weights found for the latest rebalance date.")

if latest["ticker"].duplicated().any():
    raise ValueError("Duplicate tickers found for the selected date.")

if not latest["weight"].map(lambda x: pd.notna(x) and abs(x) < float("inf")).all():
    raise ValueError("Weights contain invalid values.")

weight_sum = latest["weight"].sum()
if abs(weight_sum - 1.0) > 1e-4:
    raise ValueError(f"Weights sum to {weight_sum:.8f}, not approximately 1.")

latest["weight_pct"] = latest["weight"] * 100
latest = latest.sort_values("weight", ascending=False)
latest["rebal_date"] = latest["rebal_date"].dt.strftime("%Y-%m-%d")
latest = latest[["rebal_date", "ticker", "weight", "weight_pct"]]

out.parent.mkdir(parents=True, exist_ok=True)
latest.to_csv(out, index=False)

print(f"Rebalance date: {latest['rebal_date'].iloc[0]}")
print(f"Number of stocks: {len(latest)}")
print(f"Total allocation: {latest['weight_pct'].sum():.4f}%")
print(f"Saved to: {out}")
print("\nLargest allocations:")
print(latest[["ticker", "weight_pct"]].head(10).to_string(index=False))
