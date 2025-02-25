#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Multi-Asset Strategy Optimization

This example demonstrates how to optimize parameters for multi-asset strategies
using the backtesting.py framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import MultiAssetBacktest, MultiAssetStrategy
from backtesting.test import GOOG, AAPL, MSFT


class OptimizableSMA(MultiAssetStrategy):
    """
    A simple multi-asset strategy that trades based on SMA crossovers.
    Parameters are optimizable.
    """
    # Define parameters that will be optimized
    fast = 10  # Fast SMA period (same for all assets)
    slow = 30  # Slow SMA period (same for all assets)
    
    def init(self):
        # Create indicators for each asset
        for symbol in self.symbols:
            # Fast SMA
            setattr(self, f'{symbol}_fast_sma', 
                    self.I_for(symbol, 
                              lambda x: x.rolling(self.fast).mean(), 
                              self.data[symbol].Close))
            
            # Slow SMA
            setattr(self, f'{symbol}_slow_sma', 
                    self.I_for(symbol, 
                              lambda x: x.rolling(self.slow).mean(), 
                              self.data[symbol].Close))
    
    def next(self):
        # Check for signals for each asset
        for symbol in self.symbols:
            fast_sma = getattr(self, f'{symbol}_fast_sma')
            slow_sma = getattr(self, f'{symbol}_slow_sma')
            position = self.position_for(symbol)
            
            # Buy signal: fast SMA crosses above slow SMA
            if not position and fast_sma[-1] > slow_sma[-1] and fast_sma[-2] <= slow_sma[-2]:
                self.buy_for(symbol, size=0.3)  # Allocate 30% of available cash
            
            # Sell signal: fast SMA crosses below slow SMA
            elif position and fast_sma[-1] < slow_sma[-1] and fast_sma[-2] >= slow_sma[-2]:
                position.close()


class OptimizableMomentum(MultiAssetStrategy):
    """
    A multi-asset momentum strategy with optimizable parameters.
    """
    # Define parameters that will be optimized
    lookback = 20    # Lookback period for momentum calculation
    top_n = 2        # Number of top assets to hold
    threshold = 0.0  # Momentum threshold for buying
    
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
            if momentum > self.threshold:  # Only invest if momentum is above threshold
                position = self.position_for(symbol)
                if not position:
                    self.buy_for(symbol, size=position_size)


def optimize_sma_strategy():
    """
    Optimize the SMA strategy parameters.
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
    
    # Create the backtest instance
    bt = MultiAssetBacktest(
        data_dict=data_dict,
        strategy=OptimizableSMA,
        cash=100000,
        commission=.002,  # 0.2% commission
        margin=1.0
    )
    
    # Define parameter ranges to optimize
    fast_range = range(5, 21, 5)  # 5, 10, 15, 20
    slow_range = range(20, 61, 10)  # 20, 30, 40, 50, 60
    
    # Run optimization
    print("Optimizing SMA Strategy...")
    stats, heatmap = bt.optimize(
        fast=fast_range,
        slow=slow_range,
        maximize='Return [%]',
        return_heatmap=True
    )
    
    # Print optimization results
    print("\nBest Parameters:")
    print(f"Fast SMA: {bt._strategy.fast}")
    print(f"Slow SMA: {bt._strategy.slow}")
    print("\nPerformance Statistics:")
    print(stats.filter(regex='^[^_]'))
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.title('SMA Strategy Optimization Heatmap (Return [%])')
    sns_heatmap = plt.imshow(heatmap.values, cmap='viridis')
    plt.colorbar(sns_heatmap)
    plt.xticks(range(len(heatmap.columns)), heatmap.columns)
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    plt.xlabel('Slow SMA Period')
    plt.ylabel('Fast SMA Period')
    plt.savefig('sma_optimization_heatmap.png')
    
    # Plot the results for each asset
    for symbol in data_dict.keys():
        bt.plot(symbol=symbol, filename=f'optimized_sma_{symbol}.html')
    
    return stats, heatmap


def optimize_momentum_strategy():
    """
    Optimize the Momentum strategy parameters.
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
    
    # Create the backtest instance
    bt = MultiAssetBacktest(
        data_dict=data_dict,
        strategy=OptimizableMomentum,
        cash=100000,
        commission=.002,  # 0.2% commission
        margin=1.0
    )
    
    # Define parameter ranges to optimize
    lookback_range = range(10, 41, 10)  # 10, 20, 30, 40
    threshold_range = [-0.05, 0.0, 0.05, 0.1]  # Momentum thresholds
    
    # Define a constraint function
    def constraint(params):
        # Example constraint: lookback must be at least 15 if threshold is above 0
        if params['threshold'] > 0 and params['lookback'] < 15:
            return False
        return True
    
    # Run optimization
    print("\nOptimizing Momentum Strategy...")
    stats, heatmap = bt.optimize(
        lookback=lookback_range,
        threshold=threshold_range,
        top_n=[1, 2, 3],  # Try different numbers of top assets to hold
        maximize='Sharpe Ratio',
        constraint=constraint,
        return_heatmap=True
    )
    
    # Print optimization results
    print("\nBest Parameters:")
    print(f"Lookback Period: {bt._strategy.lookback}")
    print(f"Momentum Threshold: {bt._strategy.threshold}")
    print(f"Top N Assets: {bt._strategy.top_n}")
    print("\nPerformance Statistics:")
    print(stats.filter(regex='^[^_]'))
    
    # Plot the results for each asset
    for symbol in data_dict.keys():
        bt.plot(symbol=symbol, filename=f'optimized_momentum_{symbol}.html')
    
    return stats, heatmap


if __name__ == '__main__':
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        print("Seaborn not installed. Heatmap visualization will be basic.")
        has_seaborn = False
    
    # Optimize SMA strategy
    sma_stats, sma_heatmap = optimize_sma_strategy()
    
    # Optimize Momentum strategy
    momentum_stats, momentum_heatmap = optimize_momentum_strategy()
    
    # Compare the optimized strategies
    print("\nStrategy Comparison:")
    comparison = pd.DataFrame({
        'Optimized SMA': sma_stats.filter(regex='^[^_]'),
        'Optimized Momentum': momentum_stats.filter(regex='^[^_]')
    })
    print(comparison)
    
    # Plot comparison
    metrics = ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Win Rate [%]']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.bar(['SMA', 'Momentum'], [comparison['Optimized SMA'][metric], 
                                     comparison['Optimized Momentum'][metric]])
        plt.title(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.show() 
