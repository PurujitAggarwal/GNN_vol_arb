"""
Build HAR-RV features from daily realised volatility data.

For each stock on each day, we compute three lagged features:
    - Daily:   sqrt(RV) from yesterday
    - Weekly:  average sqrt(RV) over the past 5 days
    - Monthly: average sqrt(RV) over the past 22 days

These correspond to the standard HAR model (Corsi 2009),
using the sqrt form (Paper 1 Eq.11, Paper 2 Eq.5).
"""

import numpy as np
import pandas as pd

from gnn_vol.config import (
    HAR_DAILY_LAG,
    HAR_WEEKLY_LAG,
    HAR_MONTHLY_LAG,
)


def build_har_features(rv_df: pd.DataFrame) -> dict:
    """
    Build the HAR feature tensor from a DataFrame of sqrt(RV).

    Args:
        rv_df: DataFrame with DatetimeIndex, one column per ticker,
               values = daily sqrt(RV). Output of RVComputer.compute_rv().

    Returns:
        dict with:
            "V":       np.ndarray of shape (T, N, 3) — the feature tensor
                       where the 3 channels are [daily, weekly, monthly]
            "dates":   DatetimeIndex of length T
            "tickers": list of ticker strings of length N
    """
    tickers = list(rv_df.columns)
    n_stocks = len(tickers)

    # Forward-fill small NaN gaps (missing bars on holidays, half-days, etc.)
    rv_df = rv_df.ffill()

    # Daily: yesterday's RV
    daily = rv_df.shift(HAR_DAILY_LAG)

    # Weekly: mean of days t-2 through t-5 (4 days)
    weekly = rv_df.shift(2).rolling(window=HAR_WEEKLY_LAG - 1).mean()

    # Monthly: mean of days t-6 through t-22 (17 days)
    monthly = rv_df.shift(6).rolling(window=HAR_MONTHLY_LAG - 5).mean()

    # Find rows where all three features are available (drop NaN warmup period)
    valid_mask = daily.notna().all(axis=1) & weekly.notna().all(
        axis=1) & monthly.notna().all(axis=1)

    daily = daily[valid_mask]
    weekly = weekly[valid_mask]
    monthly = monthly[valid_mask]

    dates = daily.index
    n_days = len(dates)

    # Stack into (T, N, 3) tensor
    V = np.zeros((n_days, n_stocks, 3))
    V[:, :, 0] = daily.values
    V[:, :, 1] = weekly.values
    V[:, :, 2] = monthly.values

    return {
        "V": V,
        "dates": dates,
        "tickers": tickers,
    }


def build_targets(rv_df: pd.DataFrame, horizon: int, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Build forecast targets aligned with HAR feature dates.

    For horizon=1: target is tomorrow's sqrt(RV).
    For horizon=5: target is the sum of sqrt(RV) over the next 5 days.

    Args:
        rv_df:   same DataFrame passed to build_har_features()
        horizon: number of days ahead (1 or 5)
        dates:   the dates index returned by build_har_features()

    Returns:
        np.ndarray of shape (T, N) — the target values.
        Rows with any NaN targets are set to NaN (caller should handle).
    """
    if horizon == 1:
        targets = rv_df.shift(-1)
    else:
        # Sum of next h days of sqrt(RV)
        targets = rv_df.shift(-1).rolling(window=horizon).sum().shift(-(horizon - 1))

    # Align to the same dates as the features
    targets = targets.reindex(dates)

    return targets.values


if __name__ == "__main__":
    from gnn_vol.rv_compute import RVComputer

    computer = RVComputer([])
    rv_df = computer.load_rv()

    features = build_har_features(rv_df)
    print(f"Feature tensor shape: {features['V'].shape}")
    print(f"  T (days):   {features['V'].shape[0]}")
    print(f"  N (stocks): {features['V'].shape[1]}")
    print(f"  Features:   {features['V'].shape[2]} (daily, weekly, monthly)")
    print(
        f"  Date range: {features['dates'][0].date()} to {features['dates'][-1].date()}")

    targets_1d = build_targets(rv_df, horizon=1, dates=features["dates"])
    targets_5d = build_targets(rv_df, horizon=5, dates=features["dates"])
    print(f"\n1-day targets shape: {targets_1d.shape}")
    print(f"5-day targets shape: {targets_5d.shape}")
