# Library Helpers

Source: `backtesting/lib.py`

`backtesting.lib` contains small reusable functions and strategy helpers.

## Constants

`OHLCV_AGG`

Aggregation mapping for resampling OHLCV data:

```python
df.resample("4H", label="right").agg(OHLCV_AGG).dropna()
```

`TRADES_AGG`

Aggregation mapping for resampling trades.

## Signal Helpers

`barssince(condition, default=np.inf)`

Returns how many bars have passed since `condition` was last true.

`cross(series1, series2)`

Returns true if two series crossed in either direction on the latest bar.

`crossover(series1, series2)`

Returns true if `series1` crossed above `series2` on the latest bar.

`quantile(series, quantile=None)`

With `quantile=None`, returns the rank of the latest value against prior values. With a float from `0` to `1`, returns that quantile value.

## Plotting Helpers

`plot_heatmaps(heatmap, agg="max", ncols=3, plot_width=1200, filename="", open_browser=True)`

Plots projections of the optimization heatmap returned by:

```python
stats, heatmap = bt.optimize(..., return_heatmap=True)
```

## Statistics Helpers

`compute_stats(stats, data, trades=None, risk_free_rate=0.0)`

Recomputes result metrics, usually for a subset of trades. See [Statistics](stats.md).

## Resampling

`resample_apply(rule, func, series, *args, agg=None, **kwargs)`

Applies an indicator function to a resampled series and aligns the result back to the original index.

Inside `Strategy.init()`, the result is automatically wrapped in `Strategy.I()`.

Example:

```python
class System(Strategy):
    def init(self):
        self.daily_sma = resample_apply("D", SMA, self.data.Close, 10)
```

## Random OHLC Data

`random_ohlc_data(example_data, frac=1.0, random_state=None)`

Returns a generator that yields randomized OHLC frames with similar descriptive properties to the input data.

## Strategy Base Classes

`SignalStrategy`

Helper base class for entry and exit signal arrays. Use `set_signal(entry_size, exit_portion=None)`.

`TrailingStrategy`

Helper base class that manages trailing stop-loss levels based on ATR or a percent converted to ATR units.

`FractionalBacktest`

A `Backtest` wrapper that transforms prices and volume so whole-unit engine behavior can approximate fractional-unit trading.

`MultiBacktest`

Runs the same single-asset strategy over many independent data frames. This is different from `PortfolioBacktest`, which uses shared cash and one multi-symbol strategy.
