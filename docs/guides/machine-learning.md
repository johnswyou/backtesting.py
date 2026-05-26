# Machine Learning

Machine-learning workflows fit into Backtesting.py when predictions are transformed into indicators or signals that are available without look-ahead.

## Pattern

1. Build features from historical data only.
2. Train outside the strategy or in a strictly walk-forward manner.
3. Store predictions in the input data frame or compute them in `init()`.
4. Trade from the latest revealed prediction in `next()`.

## Example Shape

```python
class PredictionStrategy(Strategy):
    threshold = 0.01

    def init(self):
        self.pred = self.I(lambda: self.data.Prediction, name="prediction", overlay=False)

    def next(self):
        if self.pred[-1] > self.threshold:
            if not self.position:
                self.buy()
        elif self.pred[-1] < -self.threshold:
            if self.position:
                self.position.close()
            self.sell()
```

## Avoid Look-Ahead

Do not fit a model on the same future rows that the strategy is about to trade.

For honest tests:

- train on data strictly before the prediction timestamp,
- shift labels so the target is future return and the feature row is current/past data,
- keep preprocessing fit inside the training window,
- validate out of sample.

## Practical Notes

- Treat predictions as just another data column or indicator.
- Keep model training outside the per-bar loop unless the point is to simulate online retraining.
- Include realistic commissions and spread.
- Validate performance with multiple date ranges and symbols.
