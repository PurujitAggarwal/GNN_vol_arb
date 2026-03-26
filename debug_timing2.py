"""Quick diagnostic: time what happens per-date in the backtest loop."""
import time
import pandas as pd
from gnn_vol.universe import ALL_TICKERS, FORECAST_TICKERS
from gnn_vol.rv_compute import RVComputer
from gnn_vol.gnn_forecaster import GNNForecaster

print("Initialising forecaster...")
rv_computer = RVComputer(ALL_TICKERS)
forecaster = GNNForecaster(
    rv_computer=rv_computer,
    all_tickers=ALL_TICKERS,
    forecast_tickers=FORECAST_TICKERS,
    verbose=True,
)

# Simulate calling forecast on consecutive dates like the backtest does
rv_df = rv_computer.load_rv()
dates = rv_df.loc["2020-07-01":"2020-08-01"].index

print(f"\nCalling get_forecast_single on {len(dates)} dates...")
t_total = time.time()

for i, date in enumerate(dates):
    t0 = time.time()
    result = forecaster.get_forecast_single(date, "AAPL", horizon=5)
    elapsed = time.time() - t0
    if i < 5 or elapsed > 1.0:
        print(
            f"  Date {i}: {date.date()} -> {result:.4f if result else 'None'} ({elapsed:.2f}s)")

print(f"\nTotal: {time.time() - t_total:.1f}s for {len(dates)} dates")
print(f"Average: {(time.time() - t_total) / len(dates):.2f}s per date")
