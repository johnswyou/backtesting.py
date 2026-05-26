# Orders, Trades, And Positions

Source: `backtesting/backtesting.py`

## Order

Orders are created by `Strategy.buy()` and `Strategy.sell()`.

Important fields:

- `size`: positive for long, negative for short.
- `symbol`: asset symbol in portfolio mode, otherwise `None`.
- `limit`: limit price or `None`.
- `stop`: stop trigger price or `None`.
- `sl`: stop-loss price for the resulting trade.
- `tp`: take-profit price for the resulting trade.
- `tag`: user-defined value copied to the trade.
- `is_long`
- `is_short`
- `is_contingent`

Cancel an unfilled order:

```python
order.cancel()
```

All orders are good until canceled.

## Trade

A filled order becomes a `Trade`.

Important fields:

- `size`
- `symbol`
- `entry_price`
- `exit_price`
- `entry_bar`
- `exit_bar`
- `entry_time`
- `exit_time`
- `tag`
- `is_long`
- `is_short`
- `pl`
- `pl_pct`
- `value`
- `sl`
- `tp`

Close a trade:

```python
trade.close()
trade.close(portion=0.5)
```

Setting `trade.sl` or `trade.tp` modifies contingent stop-loss or take-profit orders.

## Position

`Position` is an aggregate view over active trades.

Single-asset mode:

```python
if self.position:
    self.position.close()
```

Portfolio mode:

```python
self.position["AAPL"].close()
```

Important fields:

- `size`
- `pl`
- `pl_pct`
- `is_long`
- `is_short`

`position.close(portion=1.0)` closes the selected portion of each active trade in the position.

## Size Semantics

Order size is interpreted as:

- `0 < size < 1`: fraction of available liquidity,
- `size >= 1`: absolute whole units,
- negative size: short order.

Fractional asset units are not supported by `Backtest`; use `FractionalBacktest` for transformed fractional-unit trading.

## Execution Timing

- Market orders fill on the next bar open by default.
- With `trade_on_close=True`, market orders fill on the current bar close.
- Limit and stop orders fill when their conditions are met.
- In portfolio mode, orders placed on the final bar cannot fill because there is no next bar.
