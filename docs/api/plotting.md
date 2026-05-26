# Plotting

Source: `backtesting/_plotting.py`

Plotting uses Bokeh.

## Backtest.plot

After running a single-asset backtest:

```python
stats = bt.run()
bt.plot(results=stats)
```

The plot can include:

- OHLC candles,
- equity curve,
- drawdown,
- indicators declared with `Strategy.I(..., plot=True)`,
- trade entry and exit markers.

Common options are passed through `Backtest.plot()`.

Use `open_browser=False` in tests and automated environments:

```python
bt.plot(results=stats, open_browser=False)
```

## PortfolioBacktest.plot

Portfolio plots show one symbol's candles and trades with the global portfolio equity curve:

```python
stats = bt.run()
bt.plot(symbol="AAPL", results=stats)
```

When the portfolio contains more than one symbol, `symbol` is required.

## Notebook Output

Use:

```python
from backtesting import set_bokeh_output

set_bokeh_output(notebook=True)
```

This configures Bokeh output for notebooks.

## Notes

- Very large plots may be resampled for readability.
- Indicators can be overlaid on price or plotted in separate panels.
- Plot files are HTML artifacts and are ignored by git.
