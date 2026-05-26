# API Overview

This directory documents the public API by concept instead of generated symbol listings.

Core entry points:

- [`Backtest`](backtest.md): run and optimize one strategy on one OHLCV data frame.
- [`PortfolioBacktest`](portfolio-backtest.md): run one strategy across many symbols with shared cash.
- [`Strategy`](strategy.md): base class for user strategies.
- [`Order`, `Trade`, and `Position`](orders-trades.md): trading state exposed to strategies.
- [`Statistics`](stats.md): result series, trade table, equity curve, and recomputation helpers.
- [`Plotting`](plotting.md): Bokeh plotting behavior.
- [`Library Helpers`](lib.md): helper functions and reusable strategy classes from `backtesting.lib`.

The top-level import path is:

```python
from backtesting import Backtest, PortfolioBacktest, Strategy
```

Additional helpers are imported from `backtesting.lib`:

```python
from backtesting.lib import crossover, resample_apply
```
