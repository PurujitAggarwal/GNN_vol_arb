from gnn_vol.gnn_model import train_single_model
from gnn_vol.graph_builder import GraphBuilder
from gnn_vol.universe import ALL_TICKERS
import yfinance as yf
from gnn_vol.har_features import build_har_features, build_targets
from gnn_vol.rv_compute import RVComputer
import time
import numpy as np
import pandas as pd

# Time each step
print("1. Loading RV data...")
t0 = time.time()
computer = RVComputer([])
rv_df = computer.load_rv().ffill()
print(f"   Done: {time.time()-t0:.1f}s")

print("2. Building HAR features...")
t0 = time.time()
rv_train = rv_df.iloc[-504:]
features = build_har_features(rv_train)
V = features["V"]
targets = build_targets(rv_train, horizon=5, dates=features["dates"])
print(f"   Done: {time.time()-t0:.1f}s, V shape: {V.shape}")

print("3. Downloading yfinance prices...")
t0 = time.time()
raw = yf.download(ALL_TICKERS, start="2015-01-01",
                  end="2026-12-31", auto_adjust=True, progress=False)
price_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
print(f"   Done: {time.time()-t0:.1f}s")

print("4. GLASSO (fixed alpha)...")
t0 = time.time()
price_window = price_df.loc[rv_train.index[0]:rv_train.index[-1]]
price_returns = np.log(price_window / price_window.shift(1)).dropna()
price_returns = price_returns.replace([np.inf, -np.inf], np.nan).dropna()
builder = GraphBuilder(price_returns)
graph = builder.build()
W = graph["normalised"]
print(f"   Done: {time.time()-t0:.1f}s")

print("5. Train/val split...")
split = int(len(V) * 0.75)
V_train, y_train = V[:split], targets[:split]
V_val, y_val = V[split:], targets[split:]
print(f"   Train: {V_train.shape}, Val: {V_val.shape}")

print("6. Training ONE model...")
t0 = time.time()
model = train_single_model(V_train, y_train, V_val, y_val, W, seed=0)
print(f"   Done: {time.time()-t0:.1f}s")

print("\nAll done!")
