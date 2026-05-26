# Multiple Time Frames

Use `backtesting.lib.resample_apply()` when a strategy needs indicators from a higher time frame.

Example: run on hourly data but use a daily moving average.

```python
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply


class MultiTimeFrameStrategy(Strategy):
    daily_period = 10

    def init(self):
        self.daily_sma = resample_apply(
            "D",
            SMA,
            self.data.Close,
            self.daily_period,
            plot=False,
        )

    def next(self):
        if self.data.Close[-1] > self.daily_sma[-1]:
            if not self.position:
                self.buy()
        elif self.position:
            self.position.close()
```

`resample_apply()`:

- converts the input series to the requested pandas offset rule,
- applies the function,
- aligns the result back to the original index,
- wraps the result in `Strategy.I()` when called inside `Strategy.init()`.

Use right-labeled resampling to avoid look-ahead bias. The helper does this internally.

Common rules:

- `"D"`: daily
- `"W-FRI"`: weekly ending Friday
- `"ME"`: month end
- `"4H"`: four-hour bars

The input series must have a datetime index.
