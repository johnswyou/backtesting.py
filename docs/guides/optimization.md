# Optimization

Use `Backtest.optimize()` to search strategy parameters.

## Parameterized Strategy

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

Parameters must exist as class variables before they can be passed to `run()` or `optimize()`.

## Grid Search

```python
bt = Backtest(GOOG, SmaCross)

best = bt.optimize(
    fast=range(5, 30, 5),
    slow=range(10, 80, 10),
    constraint=lambda p: p.fast < p.slow,
    maximize="SQN",
)
```

`constraint` receives an object whose attributes are the candidate parameter values.

## Randomized Grid

Limit the search with `max_tries`:

```python
best = bt.optimize(
    fast=range(5, 100),
    slow=range(10, 200),
    constraint=lambda p: p.fast < p.slow,
    max_tries=0.2,
    random_state=42,
)
```

When `max_tries` is between `0` and `1`, it is interpreted as a fraction of the admissible grid.

## Heatmaps

Return all tested parameter scores:

```python
best, heatmap = bt.optimize(
    fast=range(5, 30, 5),
    slow=range(10, 80, 10),
    constraint=lambda p: p.fast < p.slow,
    return_heatmap=True,
)
```

Plot projections:

```python
from backtesting.lib import plot_heatmaps

plot_heatmaps(heatmap)
```

## Model-Based Optimization

Use `method="sambo"` with the optional `sambo` package:

```python
best = bt.optimize(
    method="sambo",
    max_tries=200,
    fast=range(5, 100),
    slow=range(10, 200),
    constraint=lambda p: p.fast < p.slow,
)
```

## Practical Notes

- Optimize on a metric that matches the intended strategy objective.
- Keep constraints explicit.
- Use out-of-sample validation outside the optimizer.
- Be careful with indicators that have long warmup periods; different parameter sets can start trading on different bars.
