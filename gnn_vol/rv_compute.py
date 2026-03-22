

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

    def fetch_intraday(self, ticker: str, start: str, end: str) -> pd.DataFrame:

        all_bars = []
        url = f"{ALPACA_BASE_URL}/stocks/{ticker}/bars"
        params = {
            "timeframe": INTRADAY_INTERVAL,
            "start": start,
            "end": end,
            "limit": 10000,
            "adjustment": "split",
        }

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

            next_token = data.get("next_page_token")
            if next_token:
                params["page_token"] = next_token
                time.sleep(0.5)
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
        time.sleep(2)

        return df

    def _daily_rv_from_bars(self, intraday_df: pd.DataFrame) -> pd.Series:

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

        rv_dict = {}

        for ticker in self.tickers:
            intraday = self.fetch_and_cache_ticker(ticker, start, end)
            if intraday.empty:
                print(f"  [!] Skipping {ticker}: no intraday data")
                continue
            rv_dict[ticker] = self._daily_rv_from_bars(intraday)

        rv_df = pd.DataFrame(rv_dict)
        rv_df = rv_df.sort_index()

        out_path = RV_DIR / "rv_sqrt.parquet"
        rv_df.to_parquet(out_path)
        print(
            f"  Saved RV data: {rv_df.shape[0]} days x {rv_df.shape[1]} stocks -> {out_path}")

        return rv_df

    def load_rv(self) -> pd.DataFrame:

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
