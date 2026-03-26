"""
GNN-HAR volatility forecaster — main interface for the rest of the system.

Manages the full pipeline: RV data, graph construction, HAR features,
model training, and forecast generation. Outputs annualised volatility
in the same format as garch_modelling() for direct A/B comparison.

Usage:
    forecaster = GNNForecaster(all_tickers, forecast_tickers)
    sigma = forecaster.get_forecast_single(current_date, "AAPL", horizon=1)
    # sigma is annualised vol, e.g. 0.25 = 25%
"""

import numpy as np
import pandas as pd
import yfinance as yf

from gnn_vol.config import (
    TRAIN_WINDOW,
    VALIDATION_SPLIT,
    REFIT_FREQUENCY,
    ANNUALISATION_FACTOR,
    FORECAST_HORIZONS,
)
from gnn_vol.rv_compute import RVComputer
from gnn_vol.graph_builder import GraphBuilder
from gnn_vol.har_features import build_har_features, build_targets
from gnn_vol.gnn_model import train_ensemble, predict


class GNNForecaster:

    def __init__(self, rv_computer: RVComputer, all_tickers: list[str],
                 forecast_tickers: list[str], verbose: bool = True):
        """
        Args:
            rv_computer:      RVComputer instance (already initialised with all tickers)
            all_tickers:      full ~70 stock universe (for graph + model)
            forecast_tickers: subset we actually want predictions for
        """
        self.rv_computer = rv_computer
        self.all_tickers = [t.upper() for t in all_tickers]
        self.forecast_tickers = [t.upper() for t in forecast_tickers]
        self.verbose = verbose

        # Ticker index mapping for fast lookup
        self._ticker_to_idx = {t: i for i, t in enumerate(self.all_tickers)}

        # State — populated by _refit()
        self._models = {}          # horizon -> list of trained models
        self._W = None             # normalised adjacency matrix
        self._rv_df = None         # full RV dataframe
        self._price_df = None      # daily close prices for graph construction
        self._last_refit_date = None
        self._last_train_loss = None
        self._graph_sparsity = None
        self._bias_scale = {}      # horizon -> per-stock scaling factor
        self._forecast_cache = {}  # (date, horizon) -> dict of forecasts

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_forecast(self, current_date: pd.Timestamp, horizon: int = 1) -> dict:
        """
        Get annualised vol forecasts for all forecast tickers.
        Model weights are cached between refits, but features are recomputed
        each trading day so the forecast reflects the latest RV data.
        """
        self._maybe_refit(current_date, horizon)

        if horizon not in self._models or self._W is None:
            return {t: None for t in self.forecast_tickers}

        # Find the closest RV date to current_date
        rv_dates = self._rv_df.index
        valid_dates = rv_dates[rv_dates <= current_date]
        if len(valid_dates) == 0:
            return {t: None for t in self.forecast_tickers}
        rv_date = valid_dates[-1]

        # Cache by actual RV date — recompute when new RV data is available
        cache_key = (rv_date, horizon)
        if cache_key in self._forecast_cache:
            return self._forecast_cache[cache_key]

        # Build features from a small recent window (HAR needs ~22 day lookback)
        # Use only tickers that match the graph (set during _refit)
        graph_tickers = list(self._ticker_to_idx.keys())
        rv_subset = self._rv_df[graph_tickers]
        rv_recent = rv_subset.loc[:rv_date].iloc[-50:]
        features = build_har_features(rv_recent)
        V = features["V"]

        if len(V) == 0:
            return {t: None for t in self.forecast_tickers}

        # Predict using the latest day's features
        V_latest = V[-1:, :, :]
        preds = predict(self._models[horizon], V_latest, self._W)
        preds = preds[0]

        # Apply bias correction
        if horizon in self._bias_scale and self._bias_scale[horizon] is not None:
            preds = preds * self._bias_scale[horizon]

        # Convert to annualised vol
        if horizon > 1:
            preds = preds / horizon
        annualised = preds * np.sqrt(ANNUALISATION_FACTOR)

        # Map to forecast tickers
        result = {}
        for ticker in self.forecast_tickers:
            idx = self._ticker_to_idx.get(ticker)
            if idx is not None and idx < len(annualised):
                result[ticker] = float(annualised[idx])
            else:
                result[ticker] = None

        self._forecast_cache[cache_key] = result
        return result

    def get_forecast_single(self, current_date: pd.Timestamp, ticker: str,
                            horizon: int = 1) -> float | None:
        """
        Get annualised vol forecast for a single ticker.
        Same output format as garch_modelling() -> sigma_forecast_annualized.

        Args:
            current_date: the date we're forecasting from
            ticker:       e.g. "AAPL"
            horizon:      1 or 5

        Returns:
            float (annualised vol, e.g. 0.25 = 25%) or None if unavailable
        """
        forecasts = self.get_forecast(current_date, horizon)
        return forecasts.get(ticker.upper())

    def get_diagnostics(self) -> dict:
        """Return useful info about the current model state."""
        return {
            "last_refit_date": self._last_refit_date,
            "graph_sparsity": self._graph_sparsity,
            "trained_horizons": list(self._models.keys()),
            "n_ensemble": len(self._models.get(1, [])),
            "n_stocks": len(self.all_tickers),
            "n_forecast_tickers": len(self.forecast_tickers),
        }

    def force_refit(self, current_date: pd.Timestamp):
        """Manually trigger a full refit."""
        for h in FORECAST_HORIZONS:
            self._refit(current_date, h)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_refit(self, current_date: pd.Timestamp, horizon: int):
        """Check if we need to retrain. Refit if first run or enough days have passed."""
        needs_refit = (
            self._last_refit_date is None
            or horizon not in self._models
            or (current_date - self._last_refit_date).days >= REFIT_FREQUENCY
        )

        if needs_refit:
            self._refit(current_date, horizon)

    def _refit(self, current_date: pd.Timestamp, horizon: int):
        """Full refit: load data, rebuild graph, retrain model, calibrate bias."""
        if self.verbose:
            print(
                f"\n  [GNNForecaster] Refitting for horizon={horizon} at {current_date.date()}...")

        # 1. Load RV data
        self._rv_df = self.rv_computer.load_rv().ffill()

        # 2. Get training window
        rv_train = self._rv_df.loc[:current_date].iloc[-TRAIN_WINDOW:]

        if len(rv_train) < 100:
            if self.verbose:
                print(f"  [!] Not enough data for refit: {len(rv_train)} days")
            return

        if self.verbose:
            print(
                f"  Training window: {rv_train.index[0].date()} to {rv_train.index[-1].date()} ({len(rv_train)} days)")

        # 3. Rebuild graph from PRICE returns (not RV returns)
        #    This is what Paper 2 does — GLASSO on daily close price returns
        if self._price_df is None:
            if self.verbose:
                print(f"  Downloading daily prices from yfinance...")
            raw = yf.download(self.all_tickers, start="2015-01-01", end="2026-12-31",
                              auto_adjust=True, progress=False)
            self._price_df = raw["Close"] if "Close" in raw.columns.get_level_values(
                0) else raw

        price_window = self._price_df.loc[rv_train.index[0]:rv_train.index[-1]]
        price_returns = np.log(price_window / price_window.shift(1)).dropna()
        price_returns = price_returns.replace(
            [np.inf, -np.inf], np.nan).dropna()

        # Align columns to match RV data
        common_tickers = [
            t for t in rv_train.columns if t in price_returns.columns]
        price_returns = price_returns[common_tickers]

        # Drop near-constant columns that cause ill-conditioning
        col_std = price_returns.std()
        good_cols = col_std[col_std > 1e-6].index.tolist()
        price_returns = price_returns[good_cols]

        # CRITICAL: restrict RV data to the same tickers the graph uses
        # so that W (NxN) and V (T, N, 3) have the same N
        rv_train = rv_train[price_returns.columns]

        # Update ticker index mapping to match the current graph
        self._ticker_to_idx = {t: i for i,
                               t in enumerate(price_returns.columns)}

        try:
            builder = GraphBuilder(price_returns)
            graph = builder.build()
            self._W = graph["normalised"]
            self._graph_sparsity = graph["sparsity"]
        except Exception as e:
            if self.verbose:
                print(f"  [!] GLASSO failed: {e}. Reusing previous graph.")
            if self._W is None:
                if self.verbose:
                    print(f"  [!] No previous graph. Cannot refit.")
                return

        # 4. Build HAR features and targets
        features = build_har_features(rv_train)
        V = features["V"]
        targets = build_targets(
            rv_train, horizon=horizon, dates=features["dates"])

        if len(V) == 0:
            if self.verbose:
                print(f"  [!] No valid features after building HAR")
            return

        # 5. Train/val split
        split = int(len(V) * (1 - VALIDATION_SPLIT))
        V_train, y_train = V[:split], targets[:split]
        V_val, y_val = V[split:], targets[split:]

        if self.verbose:
            print(
                f"  Train: {V_train.shape[0]} days, Val: {V_val.shape[0]} days")

        # 6. Train ensemble
        models = train_ensemble(V_train, y_train, V_val, y_val, self._W)
        self._models[horizon] = models

        # 7. Compute bias correction on validation set
        #    Scale predictions so their mean matches the actual mean per stock
        val_preds = predict(models, V_val, self._W)
        valid_mask = ~np.isnan(y_val)
        val_preds_clean = np.where(valid_mask, val_preds, np.nan)
        y_val_clean = np.where(valid_mask, y_val, np.nan)

        pred_mean = np.nanmean(val_preds_clean, axis=0)
        actual_mean = np.nanmean(y_val_clean, axis=0)

        # Per-stock scale: actual / predicted (clamp to avoid extremes)
        scale = np.where(pred_mean > 1e-8, actual_mean / pred_mean, 1.0)
        scale = np.clip(scale, 0.5, 10.0)  # allow larger corrections
        self._bias_scale[horizon] = scale

        self._last_refit_date = current_date

        if self.verbose:
            avg_scale = np.mean(scale)
            print(f"  Bias correction: avg scale {avg_scale:.3f}")
            print(
                f"  [GNNForecaster] Refit complete. Graph sparsity: {self._graph_sparsity:.1f}%")


if __name__ == "__main__":
    from gnn_vol.universe import ALL_TICKERS, FORECAST_TICKERS

    # Set up
    computer = RVComputer(ALL_TICKERS)
    forecaster = GNNForecaster(computer, ALL_TICKERS, FORECAST_TICKERS)

    # Pick a date to forecast from
    test_date = pd.Timestamp("2025-01-02")

    # Get 1-day ahead forecasts
    print("\n1-day ahead forecasts:")
    forecasts_1d = forecaster.get_forecast(test_date, horizon=1)
    for ticker, vol in forecasts_1d.items():
        if vol is not None:
            print(f"  {ticker}: {vol:.4f} ({vol*100:.1f}%)")

    # Get 5-day ahead forecasts
    print("\n5-day ahead forecasts:")
    forecasts_5d = forecaster.get_forecast(test_date, horizon=5)
    for ticker, vol in forecasts_5d.items():
        if vol is not None:
            print(f"  {ticker}: {vol:.4f} ({vol*100:.1f}%)")

    # Test single ticker interface (matches garch_modelling output)
    sigma = forecaster.get_forecast_single(test_date, "AAPL", horizon=1)
    print(f"\nAAPL single forecast: {sigma:.4f} ({sigma*100:.1f}% annualised)")

    # Diagnostics
    print(f"\nDiagnostics: {forecaster.get_diagnostics()}")
