"""
diagnostics.py — GNN-HAR Vol Arb Problem Ticker Diagnostic Tool
================================================================
Analyses the 4 problem tickers (INTC, BA, AMD, UNH) from their
GNN_results trade logs and daily price data to identify:
  - When and why trades lost money
  - Vol regime state at entry
  - Forecast accuracy over time
  - Skew of winners vs losers
  - Regime detection signals

Usage:
    python3 diagnostics.py
    python3 diagnostics.py --tickers INTC BA          # specific tickers
    python3 diagnostics.py --results_dir GNN_results  # custom results dir
    python3 diagnostics.py --loss_threshold -300      # custom loss threshold
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ['INTC', 'BA', 'AMD', 'UNH']
DEFAULT_RESULTS_DIR = 'GNN_results'
LOSS_THRESHOLD = -300        # trades below this $ are "significant losers"
REGIME_SHORT_WINDOW = 20          # days for short-term RV
REGIME_LONG_WINDOW = 126         # days for long-term RV (~6 months)
REGIME_RATIO_THRESH = 1.5         # short/long RV ratio above this = stressed
# 3-month stock return below this = distress flag
STRESSED_RETURN_THRESH = -0.25
ROLLING_WINDOW_3M = 63          # trading days in ~3 months

FORECAST_COL = 'gnn_forecast'      # falls back to garch_forecast if not found

# colour palette
C_GREEN = '#1d9e75'
C_RED = '#e24b4a'
C_AMBER = '#ef9f27'
C_BLUE = '#378add'
C_GRAY = '#888780'
C_PURPLE = '#7f77dd'


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_trade_log(ticker: str, results_dir: str) -> pd.DataFrame | None:
    path = f'{results_dir}/trade_log/trade_log_{ticker.lower()}_SHORT_VOL.csv'
    if not os.path.exists(path):
        print(f"  [WARN] Trade log not found: {path}")
        return None
    df = pd.read_csv(path)

    # normalise forecast column
    if 'gnn_forecast' in df.columns:
        df = df.rename(columns={'gnn_forecast': FORECAST_COL})
    elif 'garch_forecast' in df.columns:
        df = df.rename(columns={'garch_forecast': FORECAST_COL})
    else:
        candidates = [c for c in df.columns if 'forecast' in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: FORECAST_COL})
        else:
            df[FORECAST_COL] = np.nan

    df['entry_date'] = pd.to_datetime(df['entry_date'])
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    return df


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from yfinance, compute log returns and rolling RV."""
    print(f"  Fetching price data for {ticker}...")
    raw = yf.download(ticker, start=start, end=end,
                      progress=False, auto_adjust=True)
    if raw.empty:
        print(f"  [WARN] No price data returned for {ticker}")
        return pd.DataFrame()

    px = raw[['Close']].copy()
    px.columns = ['close']
    px.index = pd.to_datetime(px.index)
    px['log_ret'] = np.log(px['close'] / px['close'].shift(1))

    # Rolling realised vol (annualised)
    px['rv_short'] = px['log_ret'].rolling(
        REGIME_SHORT_WINDOW).std() * np.sqrt(252)
    px['rv_long'] = px['log_ret'].rolling(
        REGIME_LONG_WINDOW).std() * np.sqrt(252)
    px['regime_ratio'] = px['rv_short'] / px['rv_long']

    # 3-month return for distress flag
    px['ret_3m'] = px['close'].pct_change(ROLLING_WINDOW_3M)

    # Regime label
    px['regime'] = np.where(px['regime_ratio'] >=
                            REGIME_RATIO_THRESH, 'stressed', 'normal')

    return px


def classify_trade(row) -> str:
    """Bucket a trade into a failure/success category."""
    pnl = row['net_pnl']
    exit_r = str(row.get('exit_reason', '')).lower()
    if pnl >= 500:
        return 'large_win'
    elif pnl >= 0:
        return 'small_win'
    elif pnl >= LOSS_THRESHOLD:
        return 'small_loss'
    else:
        if 'stop' in exit_r:
            return 'stop_loss'
        return 'large_loss'


def summarise_regime_at_entry(trades: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Join regime state from price data onto each trade at entry date."""
    if px.empty:
        trades['regime'] = 'unknown'
        trades['rv_short_entry'] = np.nan
        trades['rv_long_entry'] = np.nan
        trades['regime_ratio_entry'] = np.nan
        trades['ret_3m_entry'] = np.nan
        trades['distress_flag'] = False
        return trades

    px_reset = px[['rv_short', 'rv_long',
                   'regime_ratio', 'regime', 'ret_3m']].copy()
    px_reset.index.name = 'entry_date'
    px_reset = px_reset.rename(columns={
        'rv_short': 'rv_short_entry',
        'rv_long':  'rv_long_entry',
        'regime_ratio': 'regime_ratio_entry',
        'regime': 'regime',
        'ret_3m': 'ret_3m_entry',
    })

    trades = trades.copy()
    trades = trades.set_index('entry_date').join(
        px_reset, how='left').reset_index()
    trades['distress_flag'] = trades['ret_3m_entry'] < STRESSED_RETURN_THRESH
    return trades


def compute_realised_vol_during_trade(row, px: pd.DataFrame) -> float:
    """Annualised RV of the stock over the actual holding period."""
    if px.empty:
        return np.nan
    mask = (px.index >= row['entry_date']) & (px.index <= row['exit_date'])
    rets = px.loc[mask, 'log_ret'].dropna()
    if len(rets) < 2:
        return np.nan
    return rets.std() * np.sqrt(252)


# ── Per-ticker analysis ────────────────────────────────────────────────────────

def analyse_ticker(ticker: str, results_dir: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  DIAGNOSING: {ticker}")
    print(f"{'='*60}")

    trades = load_trade_log(ticker, results_dir)
    if trades is None or trades.empty:
        return None

    # date range for price download (pad by 6 months for RV warmup)
    start = (trades['entry_date'].min() -
             pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    end = (trades['exit_date'].max() +
           pd.DateOffset(days=5)).strftime('%Y-%m-%d')

    px = fetch_price_data(ticker, start, end)
    trades = summarise_regime_at_entry(trades, px)

    # Realised vol during each trade
    trades['rv_during'] = trades.apply(
        lambda r: compute_realised_vol_during_trade(r, px), axis=1
    )

    # Forecast error: how wrong was the GNN at entry
    # gnn_forecast is annualised vol %; entry_iv is also annualised
    if FORECAST_COL in trades.columns and 'entry_iv' in trades.columns:
        trades['forecast_error'] = trades[FORECAST_COL] - trades['entry_iv']
        trades['forecast_vs_rv'] = trades[FORECAST_COL] - trades['rv_during']
    else:
        trades['forecast_error'] = np.nan
        trades['forecast_vs_rv'] = np.nan

    # Trade category
    trades['category'] = trades.apply(classify_trade, axis=1)

    # Holding period
    trades['days_held'] = (trades['exit_date'] - trades['entry_date']).dt.days

    # ── Print summary ──────────────────────────────────────────────────────────

    total_pnl = trades['net_pnl'].sum()
    n_trades = len(trades)
    win_rate = (trades['net_pnl'] > 0).mean()
    avg_win = trades.loc[trades['net_pnl'] > 0, 'net_pnl'].mean()
    avg_loss = trades.loc[trades['net_pnl'] < 0, 'net_pnl'].mean()
    payoff = abs(
        avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else np.nan

    print(f"\n  OVERVIEW")
    print(f"  {'Total P&L:':<30} ${total_pnl:>10,.2f}")
    print(f"  {'Trades:':<30} {n_trades:>10}")
    print(f"  {'Win Rate:':<30} {win_rate:>10.1%}")
    print(f"  {'Avg Win:':<30} ${avg_win:>10,.2f}")
    print(f"  {'Avg Loss:':<30} ${avg_loss:>10,.2f}")
    print(f"  {'Payoff Ratio (W/L):':<30} {payoff:>10.2f}")

    # Regime breakdown
    if 'regime' in trades.columns:
        print(f"\n  REGIME BREAKDOWN AT ENTRY")
        rg = trades.groupby('regime').agg(
            count=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            win_rate=('net_pnl', lambda x: (x > 0).mean()),
            avg_pnl=('net_pnl', 'mean'),
        )
        for regime, row in rg.iterrows():
            print(f"  {regime.upper():<12} | trades: {int(row['count']):>3} | "
                  f"P&L: ${row['total_pnl']:>8,.0f} | "
                  f"win rate: {row['win_rate']:.1%} | "
                  f"avg P&L: ${row['avg_pnl']:>7,.0f}")

    # Distress flag breakdown
    if 'distress_flag' in trades.columns:
        distress_trades = trades[trades['distress_flag']]
        normal_trades = trades[~trades['distress_flag']]
        print(f"\n  DISTRESS FLAG (stock down >25% in 3 months at entry)")
        print(f"  Distressed entries: {len(distress_trades)} | "
              f"P&L: ${distress_trades['net_pnl'].sum():,.0f} | "
              f"win rate: {(distress_trades['net_pnl'] > 0).mean():.1%}")
        print(f"  Normal entries:     {len(normal_trades)} | "
              f"P&L: ${normal_trades['net_pnl'].sum():,.0f} | "
              f"win rate: {(normal_trades['net_pnl'] > 0).mean():.1%}")

    # Significant losing trades
    losers = trades[trades['net_pnl'] < LOSS_THRESHOLD].sort_values('net_pnl')
    print(f"\n  SIGNIFICANT LOSING TRADES (below ${LOSS_THRESHOLD})")
    if losers.empty:
        print("  None found.")
    else:
        cols = ['entry_date', 'exit_date', 'days_held', 'net_pnl',
                FORECAST_COL, 'entry_iv', 'rv_during', 'regime', 'distress_flag']
        cols = [c for c in cols if c in losers.columns]
        print(losers[cols].to_string(index=False))

    # Category counts
    print(f"\n  TRADE CATEGORY BREAKDOWN")
    cat_counts = trades['category'].value_counts()
    cat_pnl = trades.groupby('category')['net_pnl'].sum()
    for cat in ['large_win', 'small_win', 'small_loss', 'stop_loss', 'large_loss']:
        if cat in cat_counts:
            print(f"  {cat:<15} | count: {cat_counts[cat]:>3} | "
                  f"total P&L: ${cat_pnl.get(cat, 0):>8,.0f}")

    # Forecast accuracy
    if not trades['forecast_error'].isna().all():
        print(f"\n  FORECAST ACCURACY")
        print(f"  Mean forecast error (GNN - market IV at entry): "
              f"{trades['forecast_error'].mean():>+.1%}")
        print(f"  Mean forecast vs realised vol during trade:     "
              f"{trades['forecast_vs_rv'].mean():>+.1%}")
        print(f"  (Negative = GNN under-forecasting vol)")

    return {
        'ticker':  ticker,
        'trades':  trades,
        'px':      px,
        'total_pnl': total_pnl,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ticker_diagnostic(result: dict, output_dir: str):
    ticker = result['ticker']
    trades = result['trades']
    px = result['px']

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{ticker} — GNN-HAR Diagnostic Report',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Equity curve with regime shading ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    trades_sorted = trades.sort_values('entry_date').copy()
    trades_sorted['cum_pnl'] = trades_sorted['net_pnl'].cumsum()

    ax1.plot(trades_sorted['exit_date'], trades_sorted['cum_pnl'],
             color=C_BLUE, linewidth=2, label='Cumulative P&L')
    ax1.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.8)
    ax1.fill_between(trades_sorted['exit_date'], 0, trades_sorted['cum_pnl'],
                     where=(trades_sorted['cum_pnl'] >= 0), alpha=0.2, color=C_GREEN)
    ax1.fill_between(trades_sorted['exit_date'], 0, trades_sorted['cum_pnl'],
                     where=(trades_sorted['cum_pnl'] < 0),  alpha=0.2, color=C_RED)

    # Shade stressed regime periods
    if not px.empty and 'regime' in px.columns:
        stressed = px[px['regime'] == 'stressed']
        if not stressed.empty:
            # Find contiguous blocks
            stressed_mask = px['regime'] == 'stressed'
            blocks = []
            in_block = False
            block_start = None
            for date, val in stressed_mask.items():
                if val and not in_block:
                    in_block = True
                    block_start = date
                elif not val and in_block:
                    in_block = False
                    blocks.append((block_start, date))
            if in_block:
                blocks.append((block_start, px.index[-1]))
            for (bs, be) in blocks:
                ax1.axvspan(bs, be, alpha=0.08, color=C_RED, label='_stressed')

    # Mark significant losses
    big_losers = trades[trades['net_pnl'] < LOSS_THRESHOLD]
    if not big_losers.empty:
        cum_at_loss = []
        for _, row in big_losers.iterrows():
            mask = trades_sorted['exit_date'] <= row['exit_date']
            if mask.any():
                cum_at_loss.append(trades_sorted.loc[mask, 'cum_pnl'].iloc[-1])
            else:
                cum_at_loss.append(0)
        ax1.scatter(big_losers['exit_date'], cum_at_loss,
                    color=C_RED, zorder=5, s=60, marker='v', label='Large loss trade')

    ax1.set_ylabel('Cumulative P&L ($)', fontweight='bold')
    ax1.set_title('Cumulative P&L with Stressed Regime Shading',
                  fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    handles = [
        Patch(color=C_GREEN, alpha=0.4, label='Profit'),
        Patch(color=C_RED,   alpha=0.2, label='Stressed regime'),
        plt.Line2D([0], [0], color=C_BLUE, linewidth=2,
                   label='Cumulative P&L'),
    ]
    if not big_losers.empty:
        handles.append(plt.scatter([], [], color=C_RED,
                       marker='v', s=60, label='Large loss'))
    ax1.legend(handles=handles, loc='upper left', fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # ── 2. Regime ratio over time ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if not px.empty:
        ax2.plot(px.index, px['regime_ratio'], color=C_AMBER, linewidth=1.2)
        ax2.axhline(REGIME_RATIO_THRESH, color=C_RED, linestyle='--',
                    linewidth=1, label=f'Stress threshold ({REGIME_RATIO_THRESH}x)')
        ax2.fill_between(px.index, REGIME_RATIO_THRESH, px['regime_ratio'],
                         where=(px['regime_ratio'] >= REGIME_RATIO_THRESH),
                         alpha=0.25, color=C_RED)
        ax2.set_title('Vol Regime Ratio\n(20d RV / 126d RV)',
                      fontweight='bold')
        ax2.set_ylabel('Ratio')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=30)

    # ── 3. P&L distribution ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(trades['net_pnl'].min(), trades['net_pnl'].max(), 30)
    wins = trades[trades['net_pnl'] >= 0]['net_pnl']
    losses = trades[trades['net_pnl'] < 0]['net_pnl']
    ax3.hist(wins,   bins=bins, color=C_GREEN, alpha=0.7, label='Wins')
    ax3.hist(losses, bins=bins, color=C_RED,   alpha=0.7, label='Losses')
    ax3.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.axvline(trades['net_pnl'].mean(), color=C_BLUE, linewidth=1.5,
                linestyle=':', label=f'Mean: ${trades["net_pnl"].mean():,.0f}')
    ax3.set_title('Trade P&L Distribution', fontweight='bold')
    ax3.set_xlabel('Net P&L ($)')
    ax3.set_ylabel('Count')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle='--')

    # ── 4. Win/loss by regime ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if 'regime' in trades.columns:
        regime_groups = trades.groupby('regime').apply(
            lambda g: pd.Series({
                'wins':   (g['net_pnl'] >= 0).sum(),
                'losses': (g['net_pnl'] < 0).sum(),
            })
        )
        regimes = regime_groups.index.tolist()
        x = np.arange(len(regimes))
        w = 0.35
        ax4.bar(x - w/2, regime_groups['wins'],   w,
                color=C_GREEN, label='Wins',   alpha=0.8)
        ax4.bar(x + w/2, regime_groups['losses'], w,
                color=C_RED,   label='Losses', alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels([r.capitalize() for r in regimes])
        ax4.set_title('Win/Loss Count\nby Regime at Entry', fontweight='bold')
        ax4.set_ylabel('Number of Trades')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ── 5. Forecast vs market IV scatter ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if FORECAST_COL in trades.columns and 'entry_iv' in trades.columns:
        colors = [C_GREEN if p >= 0 else C_RED for p in trades['net_pnl']]
        ax5.scatter(trades['entry_iv'] * 100, trades[FORECAST_COL] * 100,
                    c=colors, alpha=0.7, s=40, edgecolors='none')
        # Perfect forecast line
        mn = min(trades['entry_iv'].min(), trades[FORECAST_COL].min()) * 100
        mx = max(trades['entry_iv'].max(), trades[FORECAST_COL].max()) * 100
        ax5.plot([mn, mx], [mn, mx], color=C_GRAY, linestyle='--',
                 linewidth=1, label='Perfect forecast')
        ax5.set_xlabel('Market IV at Entry (%)')
        ax5.set_ylabel('GNN Forecast (%)')
        ax5.set_title(
            'GNN Forecast vs Market IV\n(green=win, red=loss)', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, linestyle='--')

    # ── 6. Forecast error over time ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    if 'forecast_error' in trades.columns and not trades['forecast_error'].isna().all():
        trades_sorted2 = trades.sort_values('entry_date')
        rolling_err = trades_sorted2.set_index('entry_date')['forecast_error'].rolling(
            10, min_periods=1
        ).mean()
        ax6.bar(trades_sorted2['entry_date'],
                trades_sorted2['forecast_error'] * 100,
                color=[
                    C_AMBER if e < 0 else C_PURPLE for e in trades_sorted2['forecast_error']],
                alpha=0.5, width=5)
        ax6.plot(rolling_err.index, rolling_err * 100,
                 color='black', linewidth=1.5, label='10-trade rolling mean')
        ax6.axhline(0, color=C_GRAY, linestyle='--', linewidth=0.8)
        ax6.set_title(
            'Forecast Error Over Time\n(GNN forecast − market IV)', fontweight='bold')
        ax6.set_xlabel('Entry Date')
        ax6.set_ylabel('Error (pp)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.tick_params(axis='x', rotation=30)

    # ── 7. RV during trade vs forecast ────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    if 'rv_during' in trades.columns and FORECAST_COL in trades.columns:
        colors = [C_GREEN if p >= 0 else C_RED for p in trades['net_pnl']]
        ax7.scatter(trades['rv_during'] * 100, trades[FORECAST_COL] * 100,
                    c=colors, alpha=0.7, s=40, edgecolors='none')
        mn = min(trades['rv_during'].dropna().min(),
                 trades[FORECAST_COL].min()) * 100
        mx = max(trades['rv_during'].dropna().max(),
                 trades[FORECAST_COL].max()) * 100
        ax7.plot([mn, mx], [mn, mx], color=C_GRAY, linestyle='--',
                 linewidth=1, label='Perfect forecast')
        ax7.set_xlabel('Realised Vol During Trade (%)')
        ax7.set_ylabel('GNN Forecast (%)')
        ax7.set_title(
            'GNN Forecast vs\nRealised Vol During Trade', fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, linestyle='--')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'diagnostic_{ticker.lower()}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Chart saved: {out_path}")


# ── Cross-ticker summary ───────────────────────────────────────────────────────

def plot_cross_ticker_summary(all_results: list[dict], output_dir: str):
    """Single chart comparing all 4 tickers side by side."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cross-Ticker Diagnostic Summary — Problem Tickers',
                 fontsize=15, fontweight='bold')

    # ── Top left: total P&L by regime ────────────────────────────────────────
    ax = axes[0, 0]
    tickers = [r['ticker'] for r in valid]
    normal_pnl = []
    stressed_pnl = []
    for r in valid:
        t = r['trades']
        if 'regime' in t.columns:
            normal_pnl.append(t[t['regime'] == 'normal']['net_pnl'].sum())
            stressed_pnl.append(t[t['regime'] == 'stressed']['net_pnl'].sum())
        else:
            normal_pnl.append(0)
            stressed_pnl.append(0)

    x = np.arange(len(tickers))
    w = 0.35
    ax.bar(x - w/2, normal_pnl,   w, color=C_GREEN,
           alpha=0.8, label='Normal regime')
    ax.bar(x + w/2, stressed_pnl, w, color=C_RED,
           alpha=0.8, label='Stressed regime')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('P&L Split: Normal vs Stressed Regime', fontweight='bold')
    ax.set_ylabel('Total P&L ($)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # ── Top right: win rate by regime ─────────────────────────────────────────
    ax = axes[0, 1]
    normal_wr = []
    stressed_wr = []
    for r in valid:
        t = r['trades']
        if 'regime' in t.columns:
            n = t[t['regime'] == 'normal']
            s = t[t['regime'] == 'stressed']
            normal_wr.append((n['net_pnl'] > 0).mean() if len(n) > 0 else 0)
            stressed_wr.append((s['net_pnl'] > 0).mean() if len(s) > 0 else 0)
        else:
            normal_wr.append(0)
            stressed_wr.append(0)

    ax.bar(x - w/2, [v * 100 for v in normal_wr],   w,
           color=C_GREEN, alpha=0.8, label='Normal')
    ax.bar(x + w/2, [v * 100 for v in stressed_wr], w,
           color=C_RED,   alpha=0.8, label='Stressed')
    ax.axhline(50, color=C_GRAY, linestyle='--',
               linewidth=1, label='50% breakeven')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_title('Win Rate: Normal vs Stressed Regime', fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ── Bottom left: avg forecast error per ticker ────────────────────────────
    ax = axes[1, 0]
    avg_errors = []
    avg_errors_stressed = []
    for r in valid:
        t = r['trades']
        if 'forecast_error' in t.columns:
            avg_errors.append(t['forecast_error'].mean() * 100)
            if 'regime' in t.columns:
                s = t[t['regime'] == 'stressed']
                avg_errors_stressed.append(
                    s['forecast_error'].mean() * 100 if len(s) > 0 else 0
                )
            else:
                avg_errors_stressed.append(0)
        else:
            avg_errors.append(0)
            avg_errors_stressed.append(0)

    ax.bar(x - w/2, avg_errors,          w,
           color=C_PURPLE, alpha=0.8, label='All trades')
    ax.bar(x + w/2, avg_errors_stressed, w, color=C_AMBER,
           alpha=0.8, label='Stressed only')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_title('Mean Forecast Error\n(GNN − market IV, pp)',
                 fontweight='bold')
    ax.set_ylabel('Error (percentage points)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # ── Bottom right: payoff ratio comparison ─────────────────────────────────
    ax = axes[1, 1]
    payoffs_normal = []
    payoffs_stressed = []
    for r in valid:
        t = r['trades']
        for regime, store in [('normal', payoffs_normal), ('stressed', payoffs_stressed)]:
            if 'regime' in t.columns:
                sub = t[t['regime'] == regime]
            else:
                sub = t
            avg_w = sub[sub['net_pnl'] > 0]['net_pnl'].mean()
            avg_l = sub[sub['net_pnl'] < 0]['net_pnl'].mean()
            if pd.isna(avg_w) or pd.isna(avg_l) or avg_l == 0:
                store.append(0)
            else:
                store.append(abs(avg_w / avg_l))

    ax.bar(x - w/2, payoffs_normal,   w,
           color=C_GREEN, alpha=0.8, label='Normal')
    ax.bar(x + w/2, payoffs_stressed, w,
           color=C_RED,   alpha=0.8, label='Stressed')
    ax.axhline(1.0, color=C_GRAY, linestyle='--',
               linewidth=1, label='Breakeven (1:1)')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_title('Payoff Ratio (Avg Win / Avg Loss)\nby Regime',
                 fontweight='bold')
    ax.set_ylabel('Payoff Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'diagnostic_summary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nCross-ticker summary saved: {out_path}")


# ── CSV export ─────────────────────────────────────────────────────────────────

def export_annotated_logs(all_results: list[dict], output_dir: str):
    """Save annotated trade logs with regime, forecast error, rv_during columns."""
    frames = []
    for r in all_results:
        if r is not None:
            frames.append(r['trades'])
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(output_dir, 'diagnostic_annotated_trades.csv')
    combined.to_csv(out_path, index=False)
    print(f"Annotated trade log saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global LOSS_THRESHOLD

    parser = argparse.ArgumentParser(
        description='GNN-HAR problem ticker diagnostics')
    parser.add_argument('--tickers',      nargs='+', default=DEFAULT_TICKERS,
                        help='Tickers to diagnose')
    parser.add_argument('--results_dir',  default=DEFAULT_RESULTS_DIR,
                        help='Results directory containing trade_log/ subfolder')
    parser.add_argument('--loss_threshold', type=float, default=LOSS_THRESHOLD,
                        help='Net P&L threshold for "significant loss" (negative number)')
    parser.add_argument('--output_dir',   default=None,
                        help='Directory to save charts (default: <results_dir>/diagnostics/)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        args.results_dir, 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    LOSS_THRESHOLD = args.loss_threshold

    print(f"\nGNN-HAR Diagnostic Tool")
    print(f"Tickers:     {args.tickers}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Loss threshold: ${LOSS_THRESHOLD}")

    all_results = []
    for ticker in args.tickers:
        result = analyse_ticker(ticker, args.results_dir)
        if result is not None:
            plot_ticker_diagnostic(result, output_dir)
        all_results.append(result)

    # Cross-ticker summary
    print(f"\n{'='*60}")
    print("  CROSS-TICKER SUMMARY")
    print(f"{'='*60}")
    plot_cross_ticker_summary(all_results, output_dir)
    export_annotated_logs(all_results, output_dir)

    print(f"\nDone. All outputs saved to: {output_dir}/")


if __name__ == '__main__':
    main()
