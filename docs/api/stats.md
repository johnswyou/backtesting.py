# Statistics

Source: `backtesting/_stats.py` and `backtesting/lib.py`

`Backtest.run()` and `PortfolioBacktest.run()` return a pandas `Series`.

## Public Metrics

The stats series includes:

- `Start`
- `End`
- `Duration`
- `Exposure Time [%]`
- `Equity Final [$]`
- `Equity Peak [$]`
- `Return [%]`
- `Buy & Hold Return [%]`
- `Return (Ann.) [%]`
- `Volatility (Ann.) [%]`
- `CAGR [%]`
- `Sharpe Ratio`
- `Sortino Ratio`
- `Calmar Ratio`
- `Alpha [%]`
- `Beta`
- `Max. Drawdown [%]`
- `Avg. Drawdown [%]`
- `Max. Drawdown Duration`
- `Avg. Drawdown Duration`
- `# Trades`
- `Win Rate [%]`
- `Best Trade [%]`
- `Worst Trade [%]`
- `Avg. Trade [%]`
- `Max. Trade Duration`
- `Avg. Trade Duration`
- `Profit Factor`
- `Expectancy [%]`
- `SQN`
- `Kelly Criterion`

## Attached Objects

The result also includes:

- `_strategy`: the strategy instance used for the run.
- `_equity_curve`: a data frame with equity and drawdown series.
- `_trades`: a data frame of closed trades.

These fields are prefixed with `_` because they are structured objects rather than scalar display metrics.

## Trade Table

The `_trades` frame includes trade-level data such as:

- `Size`
- `EntryBar`
- `ExitBar`
- `EntryPrice`
- `ExitPrice`
- `SL`
- `TP`
- `PnL`
- `Commission`
- `ReturnPct`
- `EntryTime`
- `ExitTime`
- `Duration`
- `Tag`

For `PortfolioBacktest`, `_trades` also includes:

- `Symbol`

## Recomputing Stats

Use `backtesting.lib.compute_stats()` to recompute metrics for a subset of trades:

```python
from backtesting.lib import compute_stats

stats = bt.run()
long_trades = stats._trades[stats._trades.Size > 0]
long_stats = compute_stats(stats=stats, data=GOOG, trades=long_trades)
```

For portfolio results, pass either the original symbol-to-data mapping or a benchmark OHLC frame:

```python
portfolio_stats = compute_stats(stats=stats, data=data_by_symbol)
```

## Benchmark Notes

Single-asset backtests use the input data close series for buy-and-hold, alpha, and beta.

Portfolio backtests use an equal-weight average of normalized symbol closes, rebased after indicator warmup. This is a benchmark, not a simulated buy-and-hold portfolio with cash, margin, or order sizing.

## Related Tests

Stats behavior is covered in `TestBacktest`, `TestLib`, portfolio-specific tests, and docs tests in `backtesting/test/_test.py`.
