"""Check the raw predictions vs actual values to diagnose the scaling issue."""
import numpy as np
import pandas as pd
from gnn_vol.rv_compute import RVComputer
from gnn_vol.universe import ALL_TICKERS, FORECAST_TICKERS
from gnn_vol.har_features import build_har_features, build_targets
from gnn_vol.graph_builder import GraphBuilder
from gnn_vol.gnn_model import train_ensemble, predict
from gnn_vol.config import TRAIN_WINDOW, VALIDATION_SPLIT, ANNUALISATION_FACTOR
import yfinance as yf

computer = RVComputer([])
rv_df = computer.load_rv().ffill()

# Use a recent window
rv_train = rv_df.loc[:"2024-01-01"].iloc[-TRAIN_WINDOW:]
print(
    f"Training window: {rv_train.index[0].date()} to {rv_train.index[-1].date()}")

# Build graph from price returns
raw = yf.download(ALL_TICKERS, start="2015-01-01",
                  end="2026-12-31", auto_adjust=True, progress=False)
price_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
price_window = price_df.loc[rv_train.index[0]:rv_train.index[-1]]
price_returns = np.log(price_window / price_window.shift(1)).dropna()
price_returns = price_returns.replace([np.inf, -np.inf], np.nan).dropna()
col_std = price_returns.std()
price_returns = price_returns.loc[:, col_std > 1e-6]

builder = GraphBuilder(price_returns)
graph = builder.build()
W = graph["normalised"]

# Build features and targets for horizon=5
features = build_har_features(rv_train)
V = features["V"]
targets = build_targets(rv_train, horizon=5, dates=features["dates"])

split = int(len(V) * (1 - VALIDATION_SPLIT))
V_train, y_train = V[:split], targets[:split]
V_val, y_val = V[split:], targets[split:]

print(f"\nTraining {5} ensemble members...")
models = train_ensemble(V_train, y_train, V_val, y_val, W)

# Predict on validation set
val_preds = predict(models, V_val, W)

# Check raw values
print(f"\n=== RAW MODEL OUTPUT (validation set) ===")
print(
    f"Predictions: mean={np.nanmean(val_preds):.6f}, min={np.nanmin(val_preds):.6f}, max={np.nanmax(val_preds):.6f}")
print(
    f"Actuals:     mean={np.nanmean(y_val):.6f}, min={np.nanmin(y_val):.6f}, max={np.nanmax(y_val):.6f}")
print(f"Ratio (actual/pred): {np.nanmean(y_val) / np.nanmean(val_preds):.2f}x")

# Check per-stock for AAPL (index 0)
ticker_idx = {t: i for i, t in enumerate(rv_train.columns)}
aapl_idx = ticker_idx.get("AAPL", 0)
print(f"\n=== AAPL (index {aapl_idx}) ===")
print(f"Raw pred mean: {np.nanmean(val_preds[:, aapl_idx]):.6f}")
print(f"Actual mean:   {np.nanmean(y_val[:, aapl_idx]):.6f}")
print(
    f"Ratio: {np.nanmean(y_val[:, aapl_idx]) / np.nanmean(val_preds[:, aapl_idx]):.2f}x")

# Now annualise the way the forecaster does it
pred_daily = np.nanmean(val_preds[:, aapl_idx]) / 5  # divide sum by horizon
annualised = pred_daily * np.sqrt(252)
print(f"\nAfter /5 and *sqrt(252): {annualised:.4f} = {annualised*100:.1f}%")

# What it should be
actual_daily = np.nanmean(y_val[:, aapl_idx]) / 5
actual_ann = actual_daily * np.sqrt(252)
print(f"Actual annualised:      {actual_ann:.4f} = {actual_ann*100:.1f}%")

# With bias correction (unclamped)
bias = np.nanmean(y_val[:, aapl_idx]) / np.nanmean(val_preds[:, aapl_idx])
corrected = annualised * bias
print(
    f"\nWith bias correction ({bias:.2f}x): {corrected:.4f} = {corrected*100:.1f}%")
