# Leveraged Pairs Trading

This guide sketches the fork's leveraged cointegration pairs-trading example in Markdown form.

The strategy idea:

- trade two related assets,
- estimate the spread between them,
- enter when the spread deviates from its rolling mean,
- exit when it reverts,
- use margin to allow long/short exposure in a shared account.

## Portfolio Setup

Use `PortfolioBacktest` because the pair shares cash and margin:

```python
bt = PortfolioBacktest(
    data,
    PairsStrategy,
    cash=100_000,
    margin=0.5,
    commission=0.0005,
    hedging=True,
)
```

`margin=0.5` means 2:1 notional leverage. Lower margin means more leverage.

## Strategy Outline

```python
class PairsStrategy(Strategy):
    lookback = 60
    entry_z = 2.0
    exit_z = 0.25
    size = 0.25

    def init(self):
        a = self.data["AAA"].Close.s
        b = self.data["BBB"].Close.s
        ratio = a / b
        mean = ratio.rolling(self.lookback).mean()
        std = ratio.rolling(self.lookback).std()
        zscore = (ratio - mean) / std
        self.zscore = self.I(lambda: zscore, name="spread z-score", overlay=False)

    def next(self):
        z = self.zscore[-1]

        if z > self.entry_z:
            self.position["AAA"].close()
            self.position["BBB"].close()
            self.sell("AAA", size=self.size)
            self.buy("BBB", size=self.size)
        elif z < -self.entry_z:
            self.position["AAA"].close()
            self.position["BBB"].close()
            self.buy("AAA", size=self.size)
            self.sell("BBB", size=self.size)
        elif abs(z) < self.exit_z:
            self.position["AAA"].close()
            self.position["BBB"].close()
```

## Risk Notes

- Cointegration estimates are unstable outside the sample used to fit them.
- Leveraged pair trades can lose on both legs.
- Fractional sizes reserve buying power when orders are processed; they are not target weights.
- Same-bar order sequencing matters when both legs compete for liquidity.
- Validate with out-of-sample data and realistic commissions/spreads.

## Useful Outputs

Inspect per-symbol trades:

```python
stats._trades.groupby("Symbol").PnL.sum()
```

Inspect portfolio equity:

```python
stats._equity_curve[["Equity", "DrawdownPct"]]
```
