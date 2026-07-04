# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Portfolio (Multi-Asset) Backtesting
# ===================================
#
# This tutorial shows how to backtest strategies that trade **several assets
# from a single shared account**. If you haven't yet, first go through the
# [Quick Start User Guide](Quick Start User Guide.html) — everything there
# (indicators declared in `Strategy.init`, decisions made bar by bar in
# `Strategy.next`, orders filled by the broker on the _next_ bar) carries
# over unchanged; a single asset is simply a portfolio of one.
#
# ## Data
#
# In multi-asset mode, `Backtest` takes a **`dict` of OHLC data frames, keyed
# by asset symbol** (equivalently, a single data frame with two-level
# `(symbol, field)` columns). For this tutorial, we conjure two extra assets
# from the example Google data by applying smooth, seeded random walks, and
# have one of them start trading a few months later:

# %%
import numpy as np
import pandas as pd

from backtesting.test import GOOG


def synthetic_asset(df, seed, drift=0):
    """A plausible OHLC series: `df` times a smooth random walk."""
    rng = np.random.default_rng(seed)
    factor = np.exp(np.cumsum(drift + rng.normal(0, .01, len(df))))
    return (df[['Open', 'High', 'Low', 'Close']].mul(factor, axis=0).round(2)
            .assign(Volume=df.Volume))


data = {
    'GOOG': GOOG.iloc[:300],
    'ALPHA': synthetic_asset(GOOG.iloc[:300], seed=1, drift=.0002),
    'BETA': synthetic_asset(GOOG.iloc[60:300], seed=2, drift=-.0001),  # Lists later
}
data['BETA'].head(3)

# %% [markdown]
# The data frames don't need to share a calendar — the backtest runs on the
# **union of all timestamps**, and bars on which an asset didn't trade
# (before its listing, during a halt, a different trading calendar ...)
# simply hold NaN for that asset. Prices are never forward- or back-filled,
# and orders in a non-trading asset wait (Good 'Til Canceled) for its next
# traded bar.
#
# ## Strategy
#
# A strategy accesses per-asset data as `self.data[symbol]`, with available
# symbols listed in `self.data.symbols`. Indicators are declared per symbol —
# passing `symbol=` to `Strategy.I` associates the indicator with that asset
# for plotting and for the warm-up period — and orders are routed with
# `self.buy(symbol=...)`/`self.sell(symbol=...)`. Per-asset positions live in
# `self.positions[symbol]`.
#
# Here is the classic moving-average crossover, run as a portfolio: each
# asset trades its own signal, all sharing one account's cash and margin.

# %%
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA


class SmaCrossPortfolio(Strategy):
    fast = 10
    slow = 25

    def init(self):
        self.sma_fast = {symbol: self.I(SMA, self.data[symbol].Close, self.fast, symbol=symbol)
                         for symbol in self.data.symbols}
        self.sma_slow = {symbol: self.I(SMA, self.data[symbol].Close, self.slow, symbol=symbol)
                         for symbol in self.data.symbols}

    def next(self):
        for symbol in self.data.symbols:
            if np.isnan(self.data[symbol].Close[-1]):
                continue  # This asset isn't trading on this bar
            if crossover(self.sma_fast[symbol], self.sma_slow[symbol]):
                self.buy(symbol=symbol, size=.25)  # 25% of available liquidity
            elif crossover(self.sma_slow[symbol], self.sma_fast[symbol]) and self.positions[symbol]:
                self.positions[symbol].close()


bt = Backtest(data, SmaCrossPortfolio, cash=100_000, commission=.002, finalize_trades=True)
stats = bt.run()
stats

# %% [markdown]
# Note that the strategy skips assets that aren't trading on the current bar
# (their `Close[-1]` is NaN, and NaN comparisons are always false, i.e.
# "no signal"). In multi-asset mode, _'Buy & Hold Return [%]'_ and the
# Alpha/Beta benchmark refer to an equal-weighted, bar-by-bar-rebalanced
# basket of all the assets.
#
# The trades table says which asset each trade was in:

# %%
stats['_trades'].groupby('Symbol')[['Size', 'PnL', 'ReturnPct']].agg(
    {'Size': 'count', 'PnL': 'sum', 'ReturnPct': 'mean'}).rename(columns={'Size': 'Trades'})

# %% [markdown]
# ## Plotting
#
# Each asset gets its own chart — candlesticks drawn on the bars the asset
# actually traded, its trades, its associated indicators, and the account
# equity sampled at those bars:

# %%
bt.plot(symbol='GOOG')

# %% [markdown]
# ## Pairs trading
#
# Because all assets trade from one account, long/short strategies across
# assets need no extra machinery. Let's synthesize a cointegrated pair —
# asset `Y` tracks `X` with a mean-reverting premium — and trade the classic
# spread reversion: short the expensive leg, long the cheap one, unwind when
# the spread normalizes.

# %%
rng = np.random.default_rng(0)
spread = np.zeros(300)
for i in range(1, 300):  # An Ornstein-Uhlenbeck-ish log-premium
    spread[i] = .9 * spread[i - 1] + rng.normal(0, .01)

X = GOOG.iloc[:300]
Y = (X[['Open', 'High', 'Low', 'Close']].mul(np.exp(spread), axis=0).round(2)
     .assign(Volume=X.Volume))


class PairsTrading(Strategy):
    lookback = 20
    entry_z = 1.5
    exit_z = .3

    def init(self):
        x, y = self.data['X'].Close.s, self.data['Y'].Close.s
        log_spread = np.log(y / x)
        mean = log_spread.rolling(self.lookback).mean()
        std = log_spread.rolling(self.lookback).std()
        self.zscore = self.I(lambda: (log_spread - mean) / std, name='spread z-score')

    def next(self):
        z = self.zscore[-1]
        position = self.positions['Y']
        if not position and z > self.entry_z:    # Y expensive vs X
            self.sell(symbol='Y', size=10)
            self.buy(symbol='X', size=10)
        elif not position and z < -self.entry_z:  # Y cheap vs X
            self.buy(symbol='Y', size=10)
            self.sell(symbol='X', size=10)
        elif position and abs(z) < self.exit_z:   # Spread back to normal
            for symbol in self.data.symbols:
                self.positions[symbol].close()


bt = Backtest({'X': X, 'Y': Y}, PairsTrading, cash=20_000, finalize_trades=True)
stats = bt.run()
stats[['# Trades', 'Win Rate [%]', 'Return [%]', 'Sharpe Ratio']]

# %% [markdown]
# The z-score indicator above is _account-level_ rather than per-asset, so it
# was declared without `symbol=`; such indicators gate the warm-up of every
# asset and are shown on every `plot(symbol=...)` chart.
#
# ## Notes on semantics
#
# A few things to keep in mind, all detailed in the
# [`Backtest` documentation](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Backtest):
#
# * **One account.** Cash, equity and margin are shared. Fractional order
#   sizes (e.g. `size=.25`) refer to the account's _currently available_
#   liquidity, consumed in order-placement order.
# * **Mark to market** uses each asset's most recent traded close, so equity
#   is always computed from real, if possibly stale, prices.
# * `hedging`, `exclusive_orders` and FIFO netting apply **per symbol**.
# * `Backtest(..., commission=...)` may be a callable declaring a `symbol`
#   parameter for per-asset fee schedules.
# * For running one strategy on several instruments **independently** (each
#   with its own account, for comparison rather than as a portfolio), see
#   `backtesting.lib.MultiBacktest` instead.

# %% [markdown]
# Learn more by exploring further
# [examples](https://kernc.github.io/backtesting.py/doc/backtesting/index.html#tutorials)
# or find more framework options in the [full API reference](https://kernc.github.io/backtesting.py/doc/backtesting/index.html).
