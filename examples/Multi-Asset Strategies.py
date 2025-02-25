#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Multi-Asset Strategies

This tutorial demonstrates how to use backtesting.py to create and test
strategies that trade multiple assets simultaneously.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import MultiAssetBacktest, MultiAssetStrategy
from backtesting.test import GOOG, AAPL, MSFT


class MultiAssetSMA(MultiAssetStrategy):
    """
    A simple multi-asset strategy that trades based on SMA crossovers.
    Each asset has its own SMA parameters.
    """
    # Define parameters for each asset
    # Format: {asset_symbol}_{parameter_name}
    GOOG_fast = 10
    GOOG_slow = 30
    AAPL_fast = 15
    AAPL_slow = 40
    MSFT_fast = 20
    MSFT_slow = 50
    
    def init(self):
        # Create indicators for each asset
        # GOOG indicators
        self.goog_fast_sma = self.I_for('GOOG', 
                                        lambda x: x.rolling(self.GOOG_fast).mean(), 
                                        self.data.GOOG.Close)
        self.goog_slow_sma = self.I_for('GOOG', 
                                        lambda x: x.rolling(self.GOOG_slow).mean(), 
                                        self.data.GOOG.Close)
        
        # AAPL indicators
        self.aapl_fast_sma = self.I_for('AAPL', 
                                        lambda x: x.rolling(self.AAPL_fast).mean(), 
                                        self.data.AAPL.Close)
        self.aapl_slow_sma = self.I_for('AAPL', 
                                        lambda x: x.rolling(self.AAPL_slow).mean(), 
                                        self.data.AAPL.Close)
        
        # MSFT indicators
        self.msft_fast_sma = self.I_for('MSFT', 
                                        lambda x: x.rolling(self.MSFT_fast).mean(), 
                                        self.data.MSFT.Close)
        self.msft_slow_sma = self.I_for('MSFT', 
                                        lambda x: x.rolling(self.MSFT_slow).mean(), 
                                        self.data.MSFT.Close)
    
    def next(self):
        # Check for GOOG signals
        goog_position = self.position_for('GOOG')
        if not goog_position and self.goog_fast_sma[-1] > self.goog_slow_sma[-1]:
            # Fast SMA above slow SMA - buy signal
            self.buy_for('GOOG', size=0.3)  # Allocate 30% of available cash
        elif goog_position and self.goog_fast_sma[-1] < self.goog_slow_sma[-1]:
            # Fast SMA below slow SMA - sell signal
            goog_position.close()
        
        # Check for AAPL signals
        aapl_position = self.position_for('AAPL')
        if not aapl_position and self.aapl_fast_sma[-1] > self.aapl_slow_sma[-1]:
            # Fast SMA above slow SMA - buy signal
            self.buy_for('AAPL', size=0.3)  # Allocate 30% of available cash
        elif aapl_position and self.aapl_fast_sma[-1] < self.aapl_slow_sma[-1]:
            # Fast SMA below slow SMA - sell signal
            aapl_position.close()
        
        # Check for MSFT signals
        msft_position = self.position_for('MSFT')
        if not msft_position and self.msft_fast_sma[-1] > self.msft_slow_sma[-1]:
            # Fast SMA above slow SMA - buy signal
            self.buy_for('MSFT', size=0.3)  # Allocate 30% of available cash
        elif msft_position and self.msft_fast_sma[-1] < self.msft_slow_sma[-1]:
            # Fast SMA below slow SMA - sell signal
            msft_position.close()


class MultiAssetMomentum(MultiAssetStrategy):
    """
    A multi-asset momentum strategy that allocates capital to the best performing assets.
    """
    # Define parameters
    lookback = 20  # Lookback period for momentum calculation
    top_n = 2      # Number of top assets to hold
    
    def init(self):
        # Create momentum indicators for each asset
        for symbol in self.symbols:
            # Calculate momentum as the percentage change over the lookback period
            setattr(self, f'{symbol}_momentum', 
                    self.I_for(symbol, 
                              lambda x: x.pct_change(self.lookback), 
                              self.data[symbol].Close))
    
    def next(self):
        # Calculate momentum for each asset
        momentum_values = {}
        for symbol in self.symbols:
            momentum = getattr(self, f'{symbol}_momentum')[-1]
            momentum_values[symbol] = momentum if not np.isnan(momentum) else -np.inf
        
        # Sort assets by momentum
        sorted_assets = sorted(momentum_values.items(), key=lambda x: x[1], reverse=True)
        
        # Close positions in assets that are no longer in the top N
        top_symbols = [item[0] for item in sorted_assets[:self.top_n]]
        for symbol in self.symbols:
            position = self.position_for(symbol)
            if position and symbol not in top_symbols:
                position.close()
        
        # Open positions in the top N assets
        position_size = 1.0 / self.top_n  # Equal allocation
        for symbol, momentum in sorted_assets[:self.top_n]:
            if momentum > 0:  # Only invest in assets with positive momentum
                position = self.position_for(symbol)
                if not position:
                    self.buy_for(symbol, size=position_size)


def run_multi_asset_backtest():
    """
    Run a backtest using the multi-asset SMA strategy.
    """
    # Prepare data dictionary
    data_dict = {
        'GOOG': GOOG.copy(),
        'AAPL': AAPL.copy(),
        'MSFT': MSFT.copy()
    }
    
    # Ensure all data has the same index
    # Find the common date range
    start_date = max(df.index[0] for df in data_dict.values())
    end_date = min(df.index[-1] for df in data_dict.values())
    
    # Filter data to the common date range
    for symbol in data_dict:
        data_dict[symbol] = data_dict[symbol].loc[start_date:end_date]
    
    # Run the backtest
    bt = MultiAssetBacktest(
        data_dict=data_dict,
        strategy=MultiAssetSMA,
        cash=100000,
        commission=.002,  # 0.2% commission
        margin=1.0
    )
    
    # Run the backtest
    stats = bt.run()
    print(stats)
    
    # Plot the results for each asset
    for symbol in data_dict.keys():
        bt.plot(symbol=symbol, filename=f'multi_asset_sma_{symbol}.html')
    
    return stats


def run_momentum_backtest():
    """
    Run a backtest using the multi-asset momentum strategy.
    """
    # Prepare data dictionary
    data_dict = {
        'GOOG': GOOG.copy(),
        'AAPL': AAPL.copy(),
        'MSFT': MSFT.copy()
    }
    
    # Ensure all data has the same index
    # Find the common date range
    start_date = max(df.index[0] for df in data_dict.values())
    end_date = min(df.index[-1] for df in data_dict.values())
    
    # Filter data to the common date range
    for symbol in data_dict:
        data_dict[symbol] = data_dict[symbol].loc[start_date:end_date]
    
    # Run the backtest
    bt = MultiAssetBacktest(
        data_dict=data_dict,
        strategy=MultiAssetMomentum,
        cash=100000,
        commission=.002,  # 0.2% commission
        margin=1.0
    )
    
    # Run the backtest
    stats = bt.run()
    print(stats)
    
    # Plot the results for each asset
    for symbol in data_dict.keys():
        bt.plot(symbol=symbol, filename=f'multi_asset_momentum_{symbol}.html')
    
    return stats


if __name__ == '__main__':
    # Run the SMA strategy backtest
    print("Running Multi-Asset SMA Strategy Backtest...")
    sma_stats = run_multi_asset_backtest()
    
    # Run the Momentum strategy backtest
    print("\nRunning Multi-Asset Momentum Strategy Backtest...")
    momentum_stats = run_momentum_backtest()
    
    # Compare the strategies
    print("\nStrategy Comparison:")
    comparison = pd.DataFrame({
        'SMA Strategy': sma_stats.filter(regex='^[^_]'),
        'Momentum Strategy': momentum_stats.filter(regex='^[^_]')
    })
    print(comparison) 
