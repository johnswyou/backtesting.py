# Backtest

Source: `backtesting/backtesting.py`

`Backtest` runs one `Strategy` subclass against one OHLCV data frame.

## Constructor

```python
Backtest(
    data,
    strategy,
    *,
    cash=10_000,
    spread=0.0,
    commission=0.0,
    margin=1.0,
    trade_on_close=False,
    hedging=False,
    exclusive_orders=False,
    finalize_trades=False,
)
```

## Inputs

`data` must be a `pandas.DataFrame` with:

- `Open`
- `High`
- `Low`
- `Close`
- optional `Volume`

Additional columns are allowed and are available through `Strategy.data`.

The index should be a `DatetimeIndex`. A monotonic range index works, but datetime indexes produce better durations and plotting behavior.

`strategy` must be a `Strategy` subclass, not an instance.

## Execution

Call `run()` to simulate:

```python
stats = Backtest(GOOG, SmaCross, cash=10_000, commission=0.002).run()
```

Keyword arguments passed to `run()` override strategy class variables:

```python
stats = Backtest(GOOG, SmaCross).run(fast=10, slow=30)
```

The result is a pandas `Series`. See [Statistics](stats.md).

## Optimization

`optimize()` searches parameter combinations:

```python
best = Backtest(GOOG, SmaCross).optimize(
    fast=range(5, 30, 5),
    slow=range(10, 60, 10),
    constraint=lambda p: p.fast < p.slow,
    maximize="SQN",
)
```

Supported methods:

- `method="grid"`: exhaustive or randomized grid search.
- `method="sambo"`: model-based optimization using the optional `sambo` package.

Important options:

- `maximize`: stats key or callable.
- `max_tries`: absolute run count or fraction of grid space.
- `constraint`: filters parameter combinations.
- `return_heatmap`: returns a parameter heatmap series.
- `return_optimization`: returns raw optimizer details for `method="sambo"`.
- `random_state`: makes randomized search reproducible.

## Trading Semantics

- Market orders normally fill on the next bar open.
- With `trade_on_close=True`, market orders fill on the current bar close.
- `spread` adjusts execution price by a constant relative spread.
- `commission` is charged at entry and exit.
- `margin` is the required margin ratio, so `margin=0.02` approximates 50:1 leverage.
- With `hedging=False`, opposite-facing orders close existing trades first.
- With `exclusive_orders=True`, each new order closes the previous position.
- With `finalize_trades=True`, remaining open trades are closed on the final bar for statistics.

## Common Failures

- Missing OHLC columns raise `ValueError`.
- Missing OHLC values raise `ValueError`.
- Empty data raises `ValueError`.
- A non-`Strategy` class raises `TypeError`.
- Optimization without parameters raises `ValueError`.

## Related Tests

Behavior is covered primarily in `TestBacktest`, `TestStrategy`, `TestOptimize`, `TestPlot`, and `TestRegressions` in `backtesting/test/_test.py`.
