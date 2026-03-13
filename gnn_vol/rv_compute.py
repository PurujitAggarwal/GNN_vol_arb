"""
Fetch 5-minute intraday bars from Alpha Vantage and compute daily realised volatility.
RV is defined as the sum of squared intraday log-returns.
"""
import time
import requests
import numpy as np
import pandas as pd

from gnn_vol.config import (
    ALPHA_VANTAGE_API_KEY,
    INTRADAY_DIR,
    RV_DIR,
    INTRADAY_INTERVAL,
    API_DELAY_SECONDS,
)


class RVComputer:

    def __init__(self, tickers: list[str]):
        self.tickers = [t.upper() for t in tickers]
        INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
        RV_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_intraday(self, ticker: str, month: str) -> pd.DataFrame:
        """
        Fetch 5-min bars for one ticker for one month from Alpha Vantage.

        Args:
            ticker: e.g. "AAPL"
            month:  e.g. "2024-01"

        Returns:
            DataFrame with [open, high, low, close, volume], datetime index.
            Empty DataFrame if the request fails.
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": ticker,
            "interval": INTRADAY_INTERVAL,
            "month": month,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_API_KEY,
            "datatype": "json",
        }

        response = requests.get(
            "https://www.alphavantage.co/query", params=params)
        data = response.json()

        key = f"Time Series ({INTRADAY_INTERVAL})"
        if key not in data:
            print(f"  [!] No data for {ticker} {month}: {list(data.keys())}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[key], orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def fetch_and_cache_ticker(self, ticker: str, months: list[str]) -> pd.DataFrame:
        """
        Fetch multiple months of intraday data for one ticker.
        Skips months already cached on disk.
        """
        ticker = ticker.upper()
        frames = []

        for month in months:
            cache_path = INTRADAY_DIR / f"{ticker}_{month}.parquet"

            if cache_path.exists():
                frames.append(pd.read_parquet(cache_path))
                continue

            print(f"  Fetching {ticker} {month}...")
            df = self.fetch_intraday(ticker, month)
            if df.empty:
                continue

            df.to_parquet(cache_path)
            frames.append(df)
            time.sleep(API_DELAY_SECONDS)  # respect rate limit

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()

    def _daily_rv_from_bars(self, intraday_df: pd.DataFrame) -> pd.Series:
        """
        Compute daily realised variance from 5-min close prices.

        RV_t = sum of squared 5-min log-returns on day t.
        Returns sqrt(RV_t) based on the papers
        """
        closes = intraday_df["close"].copy()
        log_ret = np.log(closes / closes.shift(1))
        log_ret = log_ret.dropna()

        # Sum of squared returns per day
        log_ret_df = log_ret.to_frame("lr")
        log_ret_df["date"] = log_ret_df.index.date
        rv = log_ret_df.groupby("date")["lr"].apply(lambda x: np.sum(x ** 2))
        rv.index = pd.to_datetime(rv.index)

        # Return sqrt form
        return np.sqrt(rv)

    def compute_rv(self, months: list[str]) -> pd.DataFrame:
        """
        Compute daily sqrt(RV) for all tickers across the given months.

        Args:
            months: list of months like ["2023-01", "2023-02", ...]

        Returns:
            DataFrame with DatetimeIndex, one column per ticker, values = sqrt(RV).
            Saved to disk as Parquet.
        """
        rv_dict = {}

        for ticker in self.tickers:
            intraday = self.fetch_and_cache_ticker(ticker, months)
            if intraday.empty:
                print(f"  [!] Skipping {ticker}: no intraday data")
                continue
            rv_dict[ticker] = self._daily_rv_from_bars(intraday)

        rv_df = pd.DataFrame(rv_dict)
        rv_df = rv_df.sort_index()

        # Save
        out_path = RV_DIR / "rv_sqrt.parquet"
        rv_df.to_parquet(out_path)
        print(
            f"  Saved RV data: {rv_df.shape[0]} days x {rv_df.shape[1]} stocks -> {out_path}")

        return rv_df

    def load_rv(self) -> pd.DataFrame:
        """Load previously computed RV from disk."""
        path = RV_DIR / "rv_sqrt.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No RV file at {path}. Run compute_rv() first.")
        return pd.read_parquet(path)


def generate_months(start: str, end: str) -> list[str]:
    """
    Generate list of month strings between two dates.

    Args:
        start: e.g. "2020-01"
        end:   e.g. "2024-12"

    Returns:
        ["2020-01", "2020-02", ..., "2024-12"]
    """
    dates = pd.date_range(start=start + "-01", end=end + "-28", freq="MS")
    return [d.strftime("%Y-%m") for d in dates]


if __name__ == "__main__":
    # Example: compute RV for a small test
    from gnn_vol.universe import ALL_TICKERS

    months = generate_months("2024-01", "2024-03")
    computer = RVComputer(ALL_TICKERS[:3])  # just 3 tickers as a test
    rv = computer.compute_rv(months)
    print(rv.head(10))
