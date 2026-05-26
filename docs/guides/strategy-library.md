# Strategy Library

`backtesting.lib` provides small helpers and composable strategy bases.

## crossover

```python
from backtesting.lib import crossover

if crossover(self.sma_fast, self.sma_slow):
    self.buy()
```

`crossover(a, b)` returns true only when `a` crossed above `b` on the latest bar.

## SignalStrategy

`SignalStrategy` turns entry/exit arrays into orders.

```python
from backtesting.lib import SignalStrategy


class MySignals(SignalStrategy):
    def init(self):
        super().init()
        entries = self.data.Close > self.data.Close.s.rolling(20).mean()
        exits = self.data.Close < self.data.Close.s.rolling(20).mean()
        self.set_signal(entries.astype(float), exits.astype(float))

    def next(self):
        super().next()
```

Call `super().init()` and `super().next()`.

## TrailingStrategy

`TrailingStrategy` manages trailing stop-losses.

```python
from backtesting.lib import TrailingStrategy


class WithTrailingStop(TrailingStrategy):
    def init(self):
        super().init()
        self.set_trailing_sl(3)

    def next(self):
        super().next()
        if not self.position:
            self.buy()
```

You can set trailing distance by ATR multiple or approximate percent:

```python
self.set_trailing_sl(6)
self.set_trailing_pct(0.05)
```

## FractionalBacktest

Use `FractionalBacktest` when prices are too large relative to cash or the asset naturally trades in fractional units.

```python
from backtesting.lib import FractionalBacktest

bt = FractionalBacktest(BTCUSD, SmaCross, fractional_unit=1 / 100_000_000)
```

## MultiBacktest

`MultiBacktest` runs independent single-asset backtests across many data frames.

Use it for comparison across instruments. Use `PortfolioBacktest` when instruments share cash and trades interact.
