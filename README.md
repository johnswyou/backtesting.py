# Backtesting.py

Backtest trading strategies in Python.

This fork keeps documentation as plain Markdown in [docs/](docs/README.md), with an [agent guide](docs/agent-guide.md), [project map](docs/project-map.md), handwritten [API reference](docs/api/README.md), and user [guides](docs/guides/quick-start.md).

## Installation

```bash
pip install backtesting
```

For local development from this repository:

```bash
pip install -e ".[test,dev]"
```

## Usage

```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    fast = 10
    slow = 30

    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, self.fast)
        self.ma2 = self.I(SMA, price, self.slow)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.position.close()
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.position.close()
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002, exclusive_orders=True)
stats = bt.run()
bt.plot()
```

Example result shape:

```text
Start                     2004-08-19 00:00:00
End                       2013-03-01 00:00:00
Duration                   3116 days 00:00:00
Exposure Time [%]                       94.27
Equity Final [$]                     68935.12
Equity Peak [$]                      68991.22
Return [%]                             589.35
Buy & Hold Return [%]                  703.46
Return (Ann.) [%]                       25.42
Volatility (Ann.) [%]                   38.43
CAGR [%]                                16.80
Sharpe Ratio                             0.66
Sortino Ratio                            1.30
Calmar Ratio                             0.77
Alpha [%]                              450.62
Beta                                     0.02
Max. Drawdown [%]                      -33.08
Avg. Drawdown [%]                       -5.58
Max. Drawdown Duration      688 days 00:00:00
Avg. Drawdown Duration       41 days 00:00:00
# Trades                                   93
Win Rate [%]                            53.76
Best Trade [%]                          57.12
Worst Trade [%]                        -16.63
Avg. Trade [%]                           1.96
Max. Trade Duration         121 days 00:00:00
Avg. Trade Duration          32 days 00:00:00
Profit Factor                            2.13
Expectancy [%]                           6.91
SQN                                      1.78
Kelly Criterion                        0.6134
_strategy              SmaCross(fast=10,slow=30)
_equity_curve                          Equ...
_trades                       Size  EntryB...
dtype: object
```

## Documentation

- [Documentation index](docs/README.md)
- [Quick start](docs/guides/quick-start.md)
- [Backtest API](docs/api/backtest.md)
- [PortfolioBacktest API](docs/api/portfolio-backtest.md)
- [Architecture](docs/architecture.md)
- [Agent guide](docs/agent-guide.md)

Documentation is Markdown-only in this fork. There is no generated documentation build step.

## Features

- Simple strategy API
- Single-asset and shared-cash multi-asset backtesting
- Built-in parameter optimization
- Reusable strategy helpers
- Detailed statistics
- Interactive Bokeh visualizations

## Tests

```bash
python -m backtesting.test
python -m unittest backtesting.test._test.TestDocs
```

## Alternatives

See [docs/alternatives.md](docs/alternatives.md).
