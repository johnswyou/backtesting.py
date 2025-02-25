# Multi-Asset Strategy Support for backtesting.py

This extension to the backtesting.py framework enables backtesting of trading strategies across multiple assets simultaneously. It allows traders and researchers to develop and test strategies that consider relationships between different assets, implement portfolio allocation techniques, and manage risk across a diverse set of instruments.

## Key Features

- **Multi-Asset Data Handling**: Manage and synchronize price data for multiple assets with a common timeline.
- **Asset-Specific Indicators**: Create and use technical indicators specific to each asset.
- **Symbol-Based Position Management**: Track positions, orders, and trades for each asset separately.
- **Portfolio-Level Strategy Logic**: Implement strategies that consider the entire portfolio, not just individual assets.
- **Asset-Specific Visualization**: Plot backtest results for each asset individually.

## Classes Added

### MultiAssetData

A data structure that provides access to OHLCV data for multiple assets, ensuring all assets share a common timeline.

### MultiAssetStrategy

Extends the base `Strategy` class with methods specific to multi-asset strategies:

- `I_for(symbol, func, *args, **kwargs)`: Declare indicators for specific assets
- `buy_for(symbol, ...)`: Place a buy order for a specific asset
- `sell_for(symbol, ...)`: Place a sell order for a specific asset
- `position_for(symbol)`: Get the current position for a specific asset
- `symbols`: Property that returns a list of all available asset symbols

### MultiAssetBacktest

Extends the base `Backtest` class to handle multiple assets:

- Accepts a dictionary of DataFrames, one for each asset
- Ensures all assets share a common timeline
- Provides asset-specific plotting capabilities

## Usage Example

```python
from backtesting import MultiAssetBacktest, MultiAssetStrategy
from backtesting.test import GOOG, AAPL, MSFT

class MyMultiAssetStrategy(MultiAssetStrategy):
    def init(self):
        # Create indicators for each asset
        for symbol in self.symbols:
            setattr(self, f'{symbol}_sma', 
                    self.I_for(symbol, lambda x: x.rolling(20).mean(), self.data[symbol].Close))
    
    def next(self):
        for symbol in self.symbols:
            sma = getattr(self, f'{symbol}_sma')
            price = self.data[symbol].Close[-1]
            position = self.position_for(symbol)
            
            if price > sma[-1] and not position:
                self.buy_for(symbol, size=0.3)  # Allocate 30% of available cash
            elif price < sma[-1] and position:
                position.close()

# Prepare data dictionary
data_dict = {
    'GOOG': GOOG.copy(),
    'AAPL': AAPL.copy(),
    'MSFT': MSFT.copy()
}

# Ensure all data has the same index
start_date = max(df.index[0] for df in data_dict.values())
end_date = min(df.index[-1] for df in data_dict.values())
for symbol in data_dict:
    data_dict[symbol] = data_dict[symbol].loc[start_date:end_date]

# Run the backtest
bt = MultiAssetBacktest(
    data_dict=data_dict,
    strategy=MyMultiAssetStrategy,
    cash=100000,
    commission=.002
)

stats = bt.run()
print(stats)

# Plot results for each asset
for symbol in data_dict.keys():
    bt.plot(symbol=symbol, filename=f'multi_asset_{symbol}.html')
```

## Advanced Strategy Examples

See the `examples/Multi-Asset Strategies.py` file for more advanced examples, including:

1. **MultiAssetSMA**: A strategy that uses different SMA parameters for each asset
2. **MultiAssetMomentum**: A strategy that allocates capital to the best-performing assets based on momentum

## Limitations

- All assets must share a common timeline (same dates/times)
- The `optimize` method is not yet fully implemented for multi-asset strategies
- Performance may be slower compared to single-asset backtests when using many assets

## Future Enhancements

- Support for asset-specific commission structures
- Enhanced optimization capabilities for multi-asset strategies
- Improved visualization of portfolio allocation over time
- Support for correlation-based strategies 
