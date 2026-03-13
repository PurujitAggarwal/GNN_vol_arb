import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def parse_data(df, price_col="Close"):
    df = df.sort_index()
    prices = df[price_col].astype(float)
    prices = prices.ffill()
    prices = prices.dropna()
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    log_returns = log_returns - log_returns.mean()

    return log_returns

def get_log_returns(file_path):
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    log_returns = parse_data(df)

    result = adfuller(log_returns) # result is [adf statistic, p value]
    assert(result[1] < 0.05) # check that the series is stationary
    
    return log_returns