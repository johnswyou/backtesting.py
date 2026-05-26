# Strategy

Source: `backtesting/backtesting.py`

`Strategy` is the base class users subclass to define trading logic.

## Minimal Shape

```python
class SmaCross(Strategy):
    fast = 10
    slow = 30

    def init(self):
        self.sma_fast = self.I(SMA, self.data.Close, self.fast)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow)

    def next(self):
        if crossover(self.sma_fast, self.sma_slow):
            self.position.close()
            self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            self.position.close()
            self.sell()
```

## Parameters

Strategy parameters are class variables. They can be overridden by `Backtest.run()` and `Backtest.optimize()`:

```python
Backtest(GOOG, SmaCross).run(fast=5, slow=20)
```

If a parameter is passed but the strategy class does not define it, the engine raises `AttributeError`.

## Lifecycle

`init()` runs once before the simulation loop.

Use it to:

- declare indicators with `self.I(...)`,
- precompute expensive arrays,
- mutate `self.data.df` if the strategy needs derived columns.

`next()` runs once per revealed bar after indicator warmup.

Use it to:

- inspect the latest data,
- place orders,
- update stops or take-profits,
- close positions or trades.

## Data Access

Single-asset mode:

```python
self.data.Close[-1]
self.data.Close.s
self.data.df
```

Portfolio mode:

```python
self.data["AAPL"].Close[-1]
self.data["AAPL"].df
```

Data arrays are full-length in `init()` and truncated to the current bar in `next()`.

## Indicators

Declare indicators with:

```python
self.I(func, *args, name=None, plot=True, overlay=None, color=None, scatter=False, **kwargs)
```

The returned object is array-like and is automatically sliced during `next()`.

Indicator functions must return arrays with the same length as the strategy data. DataFrames are interpreted as multiple indicator series.

Leading `NaN` values delay the first call to `next()`.

## Trading Methods

```python
self.buy(symbol=None, size=.9999, limit=None, stop=None, sl=None, tp=None, tag=None)
self.sell(symbol=None, size=.9999, limit=None, stop=None, sl=None, tp=None, tag=None)
```

In single-asset mode, omit `symbol`.

In portfolio mode, pass the symbol:

```python
self.buy("AAPL", size=0.25)
```

## Runtime Properties

- `self.equity`: current account equity.
- `self.data`: current data wrapper.
- `self.position`: current position in single-asset mode; symbol-keyed accessor in portfolio mode.
- `self.orders`: pending orders.
- `self.trades`: active trades.
- `self.closed_trades`: settled trades.

## Composable Strategies

When subclassing helpers from `backtesting.lib`, call `super().init()` and `super().next()` unless the helper explicitly documents otherwise.

Related guide: [Strategy Library](../guides/strategy-library.md).
