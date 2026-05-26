# PortfolioBacktest

Source: `backtesting/backtesting.py`

`PortfolioBacktest` runs one strategy across multiple assets with shared cash, margin, orders, trades, and equity.

## Constructor

```python
PortfolioBacktest(
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
    align="inner",
)
```

## Inputs

`data` must be a mapping of symbol strings to OHLCV data frames:

```python
data = {
    "AAPL": aapl_df,
    "MSFT": msft_df,
}
```

Each frame must satisfy the same OHLCV requirements as `Backtest`.

`align="inner"` is currently the only supported alignment mode. Rows outside the common index are dropped with a warning.

## Strategy Access

Inside a strategy:

```python
close = self.data["AAPL"].Close
self.buy("AAPL", size=0.25)
self.position["AAPL"].close()
```

Useful portfolio-mode accessors:

- `self.data["SYMBOL"]`: one symbol's data.
- `self.position["SYMBOL"]`: one symbol's position.
- `self.orders`: all pending orders.
- `self.trades`: all active trades.
- `self.closed_trades`: all closed trades.
- `self.equity`: global portfolio equity.

## Spread And Commission

`spread` can be one number for all symbols or a mapping by symbol:

```python
PortfolioBacktest(data, Strategy, spread={"AAPL": 0.0001, "MSFT": 0.0002})
```

`commission` can be one value/callable for all symbols or a mapping by symbol:

```python
PortfolioBacktest(data, Strategy, commission={"AAPL": 0.001, "MSFT": (1.0, 0.0005)})
```

Mapping keys must exactly match portfolio symbols.

## Fractional Sizes

A size between `0` and `1` reserves a fraction of available portfolio buying power when the order is processed.

This is not a continuously rebalanced target weight. If several same-bar orders request more buying power than available, they are processed in order and later orders may not fill.

## Results

`run()` returns the same statistics shape as `Backtest.run()`.

Portfolio-specific details:

- `_trades` contains a `Symbol` column.
- `_equity_curve` starts at the first post-warmup trading bar.
- `_trades.EntryBar` and `_trades.ExitBar` are absolute positions in the aligned input data.
- `Buy & Hold Return [%]`, `Alpha [%]`, and `Beta` use an equal-weight average of normalized symbol closes as the benchmark.

## Plotting

Portfolio plots require a symbol when there is more than one asset:

```python
stats = bt.run()
bt.plot(symbol="AAPL", results=stats)
```

The plot shows one asset's candles and trades with the global portfolio equity curve.

## Common Failures

- Empty mapping raises `ValueError`.
- Non-string or empty symbols raise `TypeError`.
- Duplicate or missing index values raise `ValueError`.
- Non-overlapping symbol indexes raise `ValueError`.
- Unsupported `align` values raise `NotImplementedError`.
- Spread or commission mappings with missing/unknown symbols raise `KeyError`.

## Related Tests

Portfolio behavior is covered in `backtesting/test/_test.py`, especially tests around `PortfolioBacktest`, symbol attribution, shared cash, sizing, stats, and plotting.
