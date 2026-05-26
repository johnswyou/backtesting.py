# Quick Start

This guide runs a simple moving-average crossover strategy on bundled sample data.

## Install

```bash
pip install backtesting
```

For local development in this repo:

```bash
pip install -e ".[test,dev]"
```

## Minimal Strategy

```python
import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG


def SMA(values, n):
    return pd.Series(values).rolling(n).mean()


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


bt = Backtest(GOOG, SmaCross, cash=10_000, commission=0.002)
stats = bt.run()
print(stats)
bt.plot()
```

## Data Requirements

Input data is a pandas `DataFrame` with columns:

- `Open`
- `High`
- `Low`
- `Close`
- optional `Volume`

Additional columns are allowed and can be used inside the strategy.

Use a datetime index when possible:

```python
df.index = pd.to_datetime(df.index)
```

## Strategy Lifecycle

Use `init()` to precompute indicators.

Use `next()` to make decisions with only the currently revealed data.

In `next()`, `self.data.Close[-1]` is the latest visible close. This avoids accidental look-ahead behavior.

## Order Timing

Market orders fill on the next bar open by default. Use `trade_on_close=True` when you intentionally want current-bar close fills.

## Results

`run()` returns a pandas `Series` with scalar metrics and attached data:

```python
stats._equity_curve
stats._trades
stats._strategy
```

See [Statistics](../api/stats.md).

## Next Steps

- [Optimization](optimization.md)
- [Multiple Time Frames](multiple-timeframes.md)
- [Multi-Asset Portfolio](multi-asset-portfolio.md)
