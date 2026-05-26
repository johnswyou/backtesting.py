# Multi-Asset Portfolio

`PortfolioBacktest` runs one strategy across many symbols with shared cash.

This is different from `MultiBacktest`. `MultiBacktest` runs the same single-asset strategy independently over many data frames. `PortfolioBacktest` is one portfolio simulation.

## Data Shape

Pass a mapping of symbol to OHLCV data frame:

```python
data = {
    "AAA": aaa_df,
    "BBB": bbb_df,
    "CCC": ccc_df,
}
```

All frames are aligned to a common index using `align="inner"`.

## Example Strategy

```python
import pandas as pd

from backtesting import PortfolioBacktest, Strategy


def SMA(values, n):
    return pd.Series(values).rolling(n).mean()


class PortfolioSma(Strategy):
    fast = 10
    slow = 30

    def init(self):
        self.fast_ma = {}
        self.slow_ma = {}
        for symbol in self.data:
            close = self.data[symbol].Close
            self.fast_ma[symbol] = self.I(SMA, close, self.fast, name=f"{symbol} fast")
            self.slow_ma[symbol] = self.I(SMA, close, self.slow, name=f"{symbol} slow")

    def next(self):
        for symbol in self.data:
            if self.fast_ma[symbol][-1] > self.slow_ma[symbol][-1]:
                if not self.position[symbol]:
                    self.buy(symbol, size=0.25)
            elif self.position[symbol]:
                self.position[symbol].close()


bt = PortfolioBacktest(data, PortfolioSma, cash=100_000)
stats = bt.run()
```

## Shared Cash

All orders use the same account cash and margin.

A fractional `size` reserves a fraction of available portfolio buying power when the order is processed. It is not a target portfolio weight that stays rebalanced.

## Per-Symbol Costs

```python
bt = PortfolioBacktest(
    data,
    PortfolioSma,
    spread={"AAA": 0.0001, "BBB": 0.0002, "CCC": 0.0001},
    commission={"AAA": 0.001, "BBB": 0.001, "CCC": (1.0, 0.0005)},
)
```

Mapping keys must exactly match the data symbols.

## Results

Portfolio results use the same stats keys as `Backtest`.

The trade table has a `Symbol` column:

```python
stats._trades[["Symbol", "Size", "EntryTime", "ExitTime", "PnL"]]
```

Benchmark metrics use an equal-weight average of normalized symbol closes.

## Plotting

```python
bt.plot(symbol="AAA", results=stats)
```

The chart shows the selected symbol's candles and trades with the global portfolio equity curve.
