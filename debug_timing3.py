"""Time each part of the main_short_vol backtest loop."""
import time
import pandas as pd
import numpy as np
from forecast_provider import get_forecaster
from implied_vol_surface import ImpliedVolSurface
from dividend_yield import get_dividend_yield
from main_short_vol import load_options_data, get_iv_surface_for_date, get_atm_option_for_dte, get_iv_for_option

ticker = 'AAPL'
ticker_upper = ticker.upper()

print("Loading data...")
t0 = time.time()
stock_data = pd.read_csv(
    f"data/{ticker}_stock_prices_2020_2024.csv", parse_dates=['date'])
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data = stock_data.set_index('date').sort_index()
print(f"  Stock data: {time.time()-t0:.1f}s")

t0 = time.time()
options_data = load_options_data(
    f"data/{ticker}_options_2020_2024.csv", ticker=ticker_upper)
print(f"  Options data: {time.time()-t0:.1f}s ({len(options_data)} rows)")

t0 = time.time()
options_by_date = dict(tuple(options_data.groupby('date', sort=False)))
print(f"  Groupby: {time.time()-t0:.1f}s ({len(options_by_date)} dates)")

print("\nCreating GNN forecaster...")
t0 = time.time()
forecaster = get_forecaster('gnn', ticker=ticker, horizon=5, verbose=False)
print(f"  Done: {time.time()-t0:.1f}s")

# Get trading dates starting from day 127
trading_dates = sorted(options_data['date'].unique())
trading_dates = trading_dates[126:]  # skip training window
dividend_yield = get_dividend_yield(ticker_upper)

print(f"\nTiming 20 dates of the backtest loop...")
t_forecast = 0
t_iv_surface = 0
t_atm = 0
t_iv_calc = 0
t_total = time.time()

for i, current_date in enumerate(trading_dates[:20]):
    if current_date not in stock_data.index:
        continue
    spot_price = stock_data.loc[current_date, 'prc']

    # Forecast
    t0 = time.time()
    garch_forecast = forecaster.forecast(current_date)
    t_forecast += time.time() - t0

    # IV surface
    t0 = time.time()
    strikes, maturities, market_prices = get_iv_surface_for_date(
        options_data, current_date, spot_price, options_by_date=options_by_date)
    t_iv_surface += time.time() - t0

    if strikes is None:
        continue

    # IV calc
    t0 = time.time()
    iv_calc = ImpliedVolSurface(
        spot_price=spot_price, risk_free_rate=0.02,
        dividend_yield=dividend_yield,
        strikes=strikes, maturities=maturities,
        market_prices=market_prices, verbose=False)
    t_iv_calc += time.time() - t0

    # ATM option
    t0 = time.time()
    atm_option = get_atm_option_for_dte(
        options_data, current_date, spot_price,
        target_dte=45, dte_range=(30, 45),
        options_by_date=options_by_date)
    t_atm += time.time() - t0

print(f"\nResults for 20 dates:")
print(f"  Forecast:     {t_forecast:.2f}s")
print(f"  IV Surface:   {t_iv_surface:.2f}s")
print(f"  IV Calc:      {t_iv_calc:.2f}s")
print(f"  ATM Option:   {t_atm:.2f}s")
print(f"  Total:        {time.time()-t_total:.2f}s")
print(f"  Per date avg: {(time.time()-t_total)/20:.2f}s")
print(
    f"  Projected for 1258 dates: {(time.time()-t_total)/20*1258:.0f}s = {(time.time()-t_total)/20*1258/60:.1f} minutes")
