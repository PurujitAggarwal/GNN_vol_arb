"""
Fetch 5-minute intraday bars from Alpaca and compute daily realised volatility.

RV is defined as the sum of squared intraday log-returns (Andersen & Bollerslev 1998).
We store sqrt(RV) as the standard form used in the HAR literature.
"""

import time
import requests
import numpy as np
import pandas as pd

from gnn_vol.config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    INTRADAY_DIR,
    RV_DIR,
    INTRADAY_INTERVAL,
)


class RVComputer:

    def __init__(self, tickers: list[str]):
        self.tickers = [t.upper() for t in tickers]
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        INTRADAY_DIR.mkdir(parents=True, exist_ok=True)
        RV_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetch intraday data
    # ------------------------------------------------------------------

    def fetch_intraday(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch 5-min bars for one ticker between two dates from Alpaca.

        Args:
            ticker: e.g. "AAPL"
            start:  e.g. "2020-01-01"
            end:    e.g. "2024-12-31"

        Returns:
            DataFrame with [open, high, low, close, volume], datetime index.
            Empty DataFrame if the request fails.
        """
        all_bars = []
        url = f"{ALPACA_BASE_URL}/stocks/{ticker}/bars"
        params = {
            "timeframe": INTRADAY_INTERVAL,
            "start": start,
            "end": end,
            "limit": 10000,
            "adjustment": "split",
        }

        # Alpaca paginates results — keep fetching until no more pages
        while True:
            # Retry up to 3 times if connection drops
            for attempt in range(3):
                try:
                    response = requests.get(
                        url, headers=self.headers, params=params)
                    break
                except requests.exceptions.ConnectionError:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        print(
                            f"  [!] Connection error for {ticker}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(
                            f"  [!] Failed to fetch {ticker} after 3 attempts")
                        return pd.DataFrame()

            if response.status_code != 200:
                print(
                    f"  [!] Error fetching {ticker}: {response.status_code} {response.text[:200]}")
                break

            data = response.json()
            bars = data.get("bars", [])

            if not bars:
                break

            all_bars.extend(bars)

            # Check for next page
            next_token = data.get("next_page_token")
            if next_token:
                params["page_token"] = next_token
                time.sleep(0.5)  # small delay between pages
            else:
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df = df.rename(columns={"t": "timestamp", "o": "open",
                       "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.sort_index()

        return df

    def fetch_and_cache_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch intraday data for one ticker, using cache if available.
        One cache file per ticker covering the full date range.
        """
        ticker = ticker.upper()
        cache_path = INTRADAY_DIR / f"{ticker}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        print(f"  Fetching {ticker} ({start} to {end})...")
        df = self.fetch_intraday(ticker, start, end)

        if df.empty:
            print(f"  [!] No data for {ticker}")
            return df

        df.to_parquet(cache_path)
        print(f"  Cached {len(df)} bars for {ticker}")
        time.sleep(2)  # pause between tickers to avoid connection resets

        return df

    # ------------------------------------------------------------------
    # Compute realised volatility
    # ------------------------------------------------------------------

    def _daily_rv_from_bars(self, intraday_df: pd.DataFrame) -> pd.Series:
        """
        Compute daily realised variance from 5-min close prices.

        RV_t = sum of squared 5-min log-returns on day t.
        Returns sqrt(RV_t) following the standard deviation form
        used in both papers (Paper 1 Eq.1, Paper 2 Eq.4).
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

    def compute_rv(self, start: str, end: str) -> pd.DataFrame:
        """
        Compute daily sqrt(RV) for all tickers across the given date range.

        Args:
            start: e.g. "2016-01-01"
            end:   e.g. "2026-01-01"

        Returns:
            DataFrame with DatetimeIndex, one column per ticker, values = sqrt(RV).
            Saved to disk as Parquet.
        """
        rv_dict = {}

        for ticker in self.tickers:
            intraday = self.fetch_and_cache_ticker(ticker, start, end)
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


if __name__ == "__main__":
    from gnn_vol.universe import ALL_TICKERS

    computer = RVComputer(ALL_TICKERS)
    rv = computer.compute_rv(start="2016-01-01", end="2026-01-01")
    print(rv.head(10))
