# Architecture

Backtesting.py is an event-driven backtesting engine with vectorized setup.

A strategy precomputes indicators in `Strategy.init()`, then the engine reveals one bar at a time and calls `Strategy.next()` to let the strategy place, modify, or close orders.

## Single-Asset Flow

Source: `backtesting/backtesting.py`

1. `Backtest(data, StrategySubclass, ...)` validates an OHLCV `pandas.DataFrame`.
2. `Backtest.run(**params)` wraps the data in `_Data`.
3. The strategy instance is created with the broker, wrapped data, and parameter overrides.
4. `strategy.init()` runs once with full-length data.
5. Indicators declared through `strategy.I(...)` are recorded and later sliced to the current bar.
6. The engine skips indicator warmup bars.
7. For each remaining bar, broker orders are processed, then `strategy.next()` runs.
8. Closed trades and equity are converted into a stats `Series`.

The core classes are:

- `Strategy`: user extension point.
- `Backtest`: single-symbol runner.
- `_Broker`: order execution, cash, margin, commission, spread, trade state.
- `Order`, `Trade`, and `Position`: order and position model exposed to strategies.
- `_Data` and `_Array`: fast array-backed data access with pandas escape hatches.

## Portfolio Flow

Source: `PortfolioBacktest` in `backtesting/backtesting.py`

`PortfolioBacktest` runs one strategy across a mapping of symbols to OHLCV frames.

1. Input is validated as `Mapping[str, pandas.DataFrame]`.
2. Each symbol frame is validated independently.
3. Frames are aligned on a common index using `align="inner"`.
4. The strategy sees `_MultiData`, so `self.data["AAPL"].Close` returns one symbol's data.
5. Orders are placed with an explicit symbol, for example `self.buy("AAPL")`.
6. `self.position["AAPL"]` exposes a symbol-specific position.
7. Cash, margin, orders, trades, and equity are shared at the portfolio level.
8. Stats use an equal-weight normalized-close benchmark.

Important portfolio semantics:

- Fractional order sizes reserve a fraction of buying power when the order is processed.
- Same-bar fractional orders are processed first-come-first-served.
- Final-bar market orders cannot fill because there is no next bar.
- `_trades` includes a `Symbol` column.
- `_equity_curve` starts at the first post-warmup trading bar.

## Data Model

OHLCV input must include:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume` is optional and filled with `NaN` when absent.

Additional columns are preserved and exposed to strategies.

Inside strategies:

- `self.data.Close` is array-like and optimized for repeated access.
- `self.data.Close[-1]` is the most recent revealed close.
- `self.data.Close.s` gives a pandas `Series`.
- `self.data.df` gives the visible data frame.
- In portfolio mode, `self.data["SYMBOL"].Close` accesses one asset.

## Indicators

Indicators are declared with `Strategy.I(func, *args, **kwargs)`.

The indicator function must return an array with the same length as the input data, or a tuple/dataframe that can be interpreted as multiple arrays of that length.

Indicators are available at full length in `init()`, but are sliced to the current bar before each `next()` call. Rolling indicators with leading `NaN` values delay the start of the simulation.

## Orders And Trades

Orders are created through `Strategy.buy()` and `Strategy.sell()`.

An order size:

- between `0` and `1` means a fraction of available liquidity,
- greater than or equal to `1` means absolute units,
- negative values represent short orders internally.

Orders can include:

- `limit`
- `stop`
- `sl`
- `tp`
- `tag`
- `symbol` in portfolio mode

Filled orders become `Trade` objects. Active trades are available through `Strategy.trades`; settled trades are available through `Strategy.closed_trades`.

## Statistics

Stats are built in `backtesting/_stats.py`.

The returned object is a pandas `Series` containing human-readable metrics plus private data frames:

- `_strategy`
- `_equity_curve`
- `_trades`

The user-facing metrics include return, drawdown, trade counts, win rate, expectancy, SQN, and Kelly criterion. See [Statistics](api/stats.md).

## Plotting

Plotting is Bokeh-based and owned by `backtesting/_plotting.py`.

`Backtest.plot()` plots the single asset, indicators, trades, drawdown, and equity. `PortfolioBacktest.plot(symbol=...)` plots one symbol's candles/trades with the global portfolio equity curve.

## Multiprocessing

Optimization and multi-backtest helpers use multiprocessing. The top-level `backtesting.Pool` wrapper keeps Linux defaults friendly to fork-based shared-memory optimization and falls back on thread-based parallelism for spawn-only contexts.
