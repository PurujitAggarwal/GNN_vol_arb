"""
Evaluate GNN-HAR vs plain HAR baseline against actual realised volatility.

Walk-forward backtest: retrain every 21 days, predict the gap, compare both
models on MAFE, QLIKE, directional accuracy.

Run from project root: python3 -m gnn_vol.evaluate
"""

import numpy as np
import pandas as pd
import yfinance as yf

from gnn_vol.universe import ALL_TICKERS, FORECAST_TICKERS
from gnn_vol.rv_compute import RVComputer
from gnn_vol.har_features import build_har_features, build_targets
from gnn_vol.graph_builder import GraphBuilder
from gnn_vol.gnn_model import train_ensemble, predict
from gnn_vol.config import (
    TRAIN_WINDOW,
    VALIDATION_SPLIT,
    REFIT_FREQUENCY,
    ANNUALISATION_FACTOR,
)


# ------------------------------------------------------------------
# Simple HAR baseline (linear, no graph, OLS)
# ------------------------------------------------------------------

class HARBaseline:
    """
    Plain HAR model estimated by OLS. No graph, no nonlinearity.
    For each stock: RV_hat = alpha + beta_d * RV_daily + beta_w * RV_weekly + beta_m * RV_monthly
    """

    def __init__(self):
        # (N, 4) — one set of [alpha, beta_d, beta_w, beta_m] per stock
        self.weights = None

    def fit(self, V, y):
        """
        Fit OLS per stock.
        V: (T, N, 3), y: (T, N)
        """
        n_stocks = V.shape[1]
        self.weights = np.zeros((n_stocks, 4))

        for i in range(n_stocks):
            # Features for stock i: add intercept column
            X = np.column_stack([np.ones(V.shape[0]), V[:, i, :]])  # (T, 4)
            yi = y[:, i]

            # Remove NaN rows
            valid = ~np.isnan(yi) & ~np.isnan(X).any(axis=1)
            X_clean = X[valid]
            y_clean = yi[valid]

            if len(y_clean) < 10:
                continue

            # OLS: weights = (X'X)^-1 X'y
            try:
                self.weights[i] = np.linalg.lstsq(
                    X_clean, y_clean, rcond=None)[0]
            except np.linalg.LinAlgError:
                pass

    def predict(self, V):
        """
        V: (T, N, 3) -> returns (T, N) predictions
        """
        n_days, n_stocks, _ = V.shape
        preds = np.zeros((n_days, n_stocks))

        for i in range(n_stocks):
            X = np.column_stack([np.ones(n_days), V[:, i, :]])
            preds[:, i] = X @ self.weights[i]

        return np.maximum(preds, 1e-8)


# ------------------------------------------------------------------
# Metrics helper
# ------------------------------------------------------------------

def compute_metrics(pred_vals, actual_vals, results_df):
    """Compute MAFE, QLIKE, MSE, directional accuracy."""
    pred_clean = np.maximum(pred_vals, 1e-8)
    actual_clean = np.maximum(actual_vals, 1e-8)

    mafe = np.mean(np.abs(pred_vals - actual_vals))

    ratio = actual_clean / pred_clean
    qlike = np.mean(ratio - np.log(ratio) - 1)

    mse = np.mean((pred_vals - actual_vals) ** 2)

    pred_changes = results_df.groupby("ticker")["pred"].diff()
    actual_changes = results_df.groupby("ticker")["actual"].diff()
    valid = pred_changes.notna() & actual_changes.notna()
    if valid.sum() > 0:
        dir_acc = (np.sign(pred_changes[valid]) == np.sign(
            actual_changes[valid])).mean()
    else:
        dir_acc = float("nan")

    pred_ann = np.mean(pred_vals) * np.sqrt(ANNUALISATION_FACTOR)
    actual_ann = np.mean(actual_vals) * np.sqrt(ANNUALISATION_FACTOR)

    return {
        "mafe": mafe,
        "qlike": qlike,
        "mse": mse,
        "dir_acc": dir_acc,
        "pred_ann": pred_ann,
        "actual_ann": actual_ann,
    }


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def evaluate(start_date="2020-01-01", end_date="2025-12-31", horizon=1):

    print(f"{'=' * 70}")
    print(f"  GNN-HAR vs HAR Baseline Evaluation")
    print(f"  Period: {start_date} to {end_date} | Horizon: {horizon}-day")
    print(f"  Forecast tickers: {len(FORECAST_TICKERS)}")
    print(f"{'=' * 70}\n")

    # Load data
    computer = RVComputer([])
    rv_df = computer.load_rv().ffill()

    all_tickers = list(rv_df.columns)
    ticker_idx = {t: i for i, t in enumerate(all_tickers)}
    forecast_names = [t for t in FORECAST_TICKERS if t in ticker_idx]

    eval_start = pd.Timestamp(start_date)
    eval_end = pd.Timestamp(end_date)
    all_dates = rv_df.loc[eval_start:eval_end].index

    # Load daily close prices for graph construction (price returns, not RV returns)
    print("  Downloading daily prices for graph construction...")
    raw = yf.download(list(rv_df.columns), start="2015-01-01", end="2026-12-31",
                      auto_adjust=True, progress=False)
    price_df = raw["Close"] if "Close" in raw.columns.get_level_values(
        0) else raw

    # Storage: one list per model
    gnn_records = []
    har_records = []
    hybrid_records = []  # GNN for connected stocks, HAR for peripheral

    refit_dates = all_dates[::REFIT_FREQUENCY]
    print(f"  Refit points: {len(refit_dates)}\n")

    gnn_models = None
    W = None
    A = None
    node_degrees = None
    har_model = None
    bias_scale = None
    MIN_DEGREE = 5  # stocks with fewer connections use HAR instead of GNN

    for refit_i, refit_date in enumerate(refit_dates):
        print(
            f"  [{refit_i + 1}/{len(refit_dates)}] Refitting at {refit_date.date()}...")

        # Training window
        rv_train = rv_df.loc[:refit_date].iloc[-TRAIN_WINDOW:]
        if len(rv_train) < 200:
            print(f"    Skipping: only {len(rv_train)} days")
            continue

        # Graph from PRICE returns (not RV returns) — matches Paper 2
        price_window = price_df.loc[rv_train.index[0]:rv_train.index[-1]]
        price_returns = np.log(price_window / price_window.shift(1)).dropna()
        price_returns = price_returns.replace(
            [np.inf, -np.inf], np.nan).dropna()
        common_tickers = [
            t for t in rv_train.columns if t in price_returns.columns]
        price_returns = price_returns[common_tickers]

        # Drop near-constant columns that cause ill-conditioning
        col_std = price_returns.std()
        price_returns = price_returns.loc[:, col_std > 1e-6]

        try:
            builder = GraphBuilder(price_returns)
            graph = builder.build()
            W = graph["normalised"]
            A = graph["adjacency"]
            # Node degrees: how many connections each stock has
            node_degrees = A.sum(axis=1)
        except Exception as e:
            print(f"    GLASSO failed: {e}. Reusing previous graph.")
            if W is None:
                print(f"    No previous graph available. Skipping.")
                continue

        # HAR features + targets
        features = build_har_features(rv_train)
        V = features["V"]
        targets = build_targets(
            rv_train, horizon=horizon, dates=features["dates"])

        if len(V) == 0:
            print(f"    Skipping: no valid features")
            continue

        # Split
        split = int(len(V) * (1 - VALIDATION_SPLIT))
        V_train, y_train = V[:split], targets[:split]
        V_val, y_val = V[split:], targets[split:]

        # Train GNN
        gnn_models = train_ensemble(V_train, y_train, V_val, y_val, W)

        # Compute bias correction on validation set
        val_preds = predict(gnn_models, V_val, W)
        valid_mask = ~np.isnan(y_val)
        val_preds_clean = np.where(valid_mask, val_preds, np.nan)
        y_val_clean = np.where(valid_mask, y_val, np.nan)
        pred_mean = np.nanmean(val_preds_clean, axis=0)
        actual_mean = np.nanmean(y_val_clean, axis=0)
        bias_scale = np.where(pred_mean > 1e-8, actual_mean / pred_mean, 1.0)
        bias_scale = np.clip(bias_scale, 0.5, 2.0)

        # Train HAR baseline
        har_model = HARBaseline()
        har_model.fit(V_train, y_train)

        # Predict forward
        predict_end_idx = min(
            all_dates.get_loc(refit_date) + REFIT_FREQUENCY,
            len(all_dates)
        )
        predict_dates = all_dates[all_dates.get_loc(
            refit_date):predict_end_idx]

        for pred_date in predict_dates:
            rv_slice = rv_df.loc[:pred_date]
            feat = build_har_features(rv_slice)
            V_curr = feat["V"]

            if len(V_curr) == 0:
                continue

            V_latest = V_curr[-1:, :, :]

            # GNN prediction (with bias correction)
            gnn_preds = predict(gnn_models, V_latest, W)[0]
            if bias_scale is not None:
                gnn_preds = gnn_preds * bias_scale

            # HAR prediction
            har_preds = har_model.predict(V_latest)[0]

            # Actual
            if horizon == 1:
                target_idx = rv_df.index.get_loc(pred_date) + 1
            else:
                target_idx = rv_df.index.get_loc(pred_date) + horizon

            if target_idx >= len(rv_df):
                continue

            actuals = rv_df.iloc[target_idx].values

            for ticker in forecast_names:
                idx = ticker_idx[ticker]
                gnn_val = gnn_preds[idx]
                har_val = har_preds[idx]
                actual_val = actuals[idx]

                if np.isnan(gnn_val) or np.isnan(har_val) or np.isnan(actual_val):
                    continue

                if horizon > 1:
                    gnn_val = gnn_val / horizon
                    har_val = har_val / horizon

                # Hybrid: use GNN if stock has enough connections, otherwise HAR
                if node_degrees is not None and idx < len(node_degrees) and node_degrees[idx] >= MIN_DEGREE:
                    hybrid_val = gnn_val
                else:
                    hybrid_val = har_val

                gnn_records.append(
                    {"date": pred_date, "ticker": ticker, "pred": gnn_val, "actual": actual_val})
                har_records.append(
                    {"date": pred_date, "ticker": ticker, "pred": har_val, "actual": actual_val})
                hybrid_records.append(
                    {"date": pred_date, "ticker": ticker, "pred": hybrid_val, "actual": actual_val})

    # Build results
    gnn_df = pd.DataFrame(gnn_records)
    har_df = pd.DataFrame(har_records)
    hybrid_df = pd.DataFrame(hybrid_records)

    if gnn_df.empty:
        print("No predictions generated.")
        return

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {len(gnn_df)} prediction pairs per model")
    print(f"{'=' * 70}")

    # Overall comparison — all three models
    gnn_metrics = compute_metrics(
        gnn_df["pred"].values, gnn_df["actual"].values, gnn_df)
    har_metrics = compute_metrics(
        har_df["pred"].values, har_df["actual"].values, har_df)
    hybrid_metrics = compute_metrics(
        hybrid_df["pred"].values, hybrid_df["actual"].values, hybrid_df)

    print(f"\n  OVERALL COMPARISON")
    print(f"  {'─' * 70}")
    print(
        f"  {'Metric':<20} {'GNN-HAR':>12} {'HAR':>12} {'HYBRID':>12} {'Hybrid vs HAR':>14}")
    print(f"  {'─' * 70}")

    for key, label in [("mafe", "MAFE"), ("qlike", "QLIKE"), ("mse", "MSE"), ("dir_acc", "Dir Accuracy")]:
        gnn_v = gnn_metrics[key]
        har_v = har_metrics[key]
        hyb_v = hybrid_metrics[key]

        if key == "dir_acc":
            imp = ((hyb_v - har_v) / har_v * 100) if har_v != 0 else 0
            print(
                f"  {label:<20} {gnn_v:>11.2%} {har_v:>11.2%} {hyb_v:>11.2%} {imp:>+12.1f}%")
        else:
            imp = ((har_v - hyb_v) / har_v * 100) if har_v != 0 else 0
            print(
                f"  {label:<20} {gnn_v:>12.6f} {har_v:>12.6f} {hyb_v:>12.6f} {imp:>+12.1f}%")

    print(f"  {'─' * 70}")
    print(
        f"  {'Mean Pred (ann)':<20} {gnn_metrics['pred_ann']:>11.1%} {har_metrics['pred_ann']:>11.1%} {hybrid_metrics['pred_ann']:>11.1%}")
    print(
        f"  {'Mean Actual (ann)':<20} {gnn_metrics['actual_ann']:>11.1%} {har_metrics['actual_ann']:>11.1%} {hybrid_metrics['actual_ann']:>11.1%}")

    # Per-ticker comparison — hybrid vs HAR
    print(f"\n  PER-TICKER: HYBRID vs HAR (MAFE)")
    print(f"  {'─' * 70}")
    print(f"  {'Ticker':<8} {'Hybrid':>10} {'HAR':>10} {'Winner':>8} {'Improv':>10} {'Method':>8} {'Pred Ann':>9} {'Act Ann':>9}")
    print(f"  {'─' * 70}")

    hybrid_wins = 0
    for ticker in forecast_names:
        hyb_mask = hybrid_df["ticker"] == ticker
        har_mask = har_df["ticker"] == ticker

        hyb_t = hybrid_df[hyb_mask]
        har_t = har_df[har_mask]

        if len(hyb_t) == 0:
            continue

        hyb_mafe = np.mean(
            np.abs(hyb_t["pred"].values - hyb_t["actual"].values))
        har_mafe = np.mean(
            np.abs(har_t["pred"].values - har_t["actual"].values))

        winner = "HYBRID" if hyb_mafe < har_mafe else "HAR"
        if winner == "HYBRID":
            hybrid_wins += 1

        imp = (har_mafe - hyb_mafe) / har_mafe * 100

        # Show which method the hybrid is using for this ticker
        idx = ticker_idx.get(ticker, 0)
        method = "GNN" if (node_degrees is not None and idx < len(
            node_degrees) and node_degrees[idx] >= MIN_DEGREE) else "HAR"

        hyb_ann = np.mean(hyb_t["pred"].values) * np.sqrt(ANNUALISATION_FACTOR)
        act_ann = np.mean(hyb_t["actual"].values) * \
            np.sqrt(ANNUALISATION_FACTOR)

        print(f"  {ticker:<8} {hyb_mafe:>10.6f} {har_mafe:>10.6f} {winner:>8} {imp:>+9.1f}% {method:>8} {hyb_ann:>8.1%} {act_ann:>8.1%}")

    print(f"  {'─' * 70}")
    print(f"  Hybrid wins: {hybrid_wins}/{len(forecast_names)} tickers")

    # Save
    gnn_df["model"] = "GNN-HAR"
    har_df["model"] = "HAR"
    hybrid_df["model"] = "HYBRID"
    combined = pd.concat([gnn_df, har_df, hybrid_df])
    out_path = "gnn_vol/data/eval_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    return gnn_df, har_df, hybrid_df


if __name__ == "__main__":
    evaluate(start_date="2020-01-01", end_date="2025-12-31", horizon=1)
    evaluate(start_date="2020-01-01", end_date="2025-12-31", horizon=5)
