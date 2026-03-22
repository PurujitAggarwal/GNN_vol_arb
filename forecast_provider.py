"""
Unified forecast provider — switches between EGARCH and GNN-HAR forecasts.

Usage in any strategy file:
    from forecast_provider import get_forecaster

    forecaster = get_forecaster(method="egarch", stock_data=stock_data, ...)
    # or
    forecaster = get_forecaster(method="gnn", ticker="AAPL", ...)

    forecast = forecaster.forecast(current_date)
"""

import pandas as pd


class EGARCHForecaster:
    """Wraps the existing VolForecaster to match the unified interface."""

    def __init__(self, stock_data, train_window=126, refit_frequency=21, verbose=False):
        from volForecaster import VolForecaster
        self._forecaster = VolForecaster(
            stock_data, train_window, refit_frequency, verbose)
        self.method = "EGARCH"

    def forecast(self, current_date):
        return self._forecaster.get_forecast(current_date)

    def record_market_iv(self, market_iv):
        self._forecaster.record_market_iv(market_iv)

    def get_diagnostics(self):
        return self._forecaster.get_forecast_diagnostics()


class GNNForecasterWrapper:
    """Wraps the GNN-HAR forecaster to match the unified interface."""

    def __init__(self, ticker, horizon=5, verbose=False):
        from gnn_vol.universe import ALL_TICKERS, FORECAST_TICKERS
        from gnn_vol.rv_compute import RVComputer
        from gnn_vol.gnn_forecaster import GNNForecaster

        self.ticker = ticker.upper()
        self.horizon = horizon
        self.method = "GNN-HAR"

        rv_computer = RVComputer(ALL_TICKERS)
        self._forecaster = GNNForecaster(
            rv_computer=rv_computer,
            all_tickers=ALL_TICKERS,
            forecast_tickers=FORECAST_TICKERS,
            verbose=verbose,
        )

    def forecast(self, current_date):
        result = self._forecaster.get_forecast_single(
            pd.Timestamp(current_date), self.ticker, horizon=self.horizon
        )
        return result

    def record_market_iv(self, market_iv):
        pass  # GNN doesn't use market IV feedback

    def get_diagnostics(self):
        return self._forecaster.get_diagnostics()


def get_forecaster(method="egarch", **kwargs):
    """
    Factory function. Returns a forecaster with a .forecast(date) method.

    For EGARCH:
        get_forecaster("egarch", stock_data=df, train_window=126, refit_frequency=21)

    For GNN:
        get_forecaster("gnn", ticker="AAPL", horizon=5)
    """
    method = method.lower()

    if method == "egarch":
        return EGARCHForecaster(
            stock_data=kwargs["stock_data"],
            train_window=kwargs.get("train_window", 126),
            refit_frequency=kwargs.get("refit_frequency", 21),
            verbose=kwargs.get("verbose", False),
        )

    elif method in ("gnn", "gnn-har", "gnnhar"):
        return GNNForecasterWrapper(
            ticker=kwargs["ticker"],
            horizon=kwargs.get("horizon", 5),
            verbose=kwargs.get("verbose", False),
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'egarch' or 'gnn'.")
