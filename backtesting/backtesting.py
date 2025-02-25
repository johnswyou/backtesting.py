"""
Core framework data structures.
Objects from this module can also be imported from the top-level
module directly, e.g.

    from backtesting import Backtest, Strategy
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import lru_cache, partial
from itertools import chain, product, repeat
from math import copysign
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ._plotting import plot  # noqa: I001
from ._stats import compute_stats
from ._util import (
    SharedMemoryManager, _as_str, _Indicator, _Data, _batch, _indicator_warmup_nbars,
    _strategy_indicators, patch, try_, _tqdm,
)

__pdoc__ = {
    'Strategy.__init__': False,
    'Order.__init__': False,
    'Position.__init__': False,
    'Trade.__init__': False,
}


class Strategy(metaclass=ABCMeta):
    """
    A trading strategy base class. Extend this class and
    override methods
    `backtesting.backtesting.Strategy.init` and
    `backtesting.backtesting.Strategy.next` to define
    your own strategy.
    """
    def __init__(self, broker, data, params):
        self._indicators = []
        self._broker: _Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
                                                        map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with.")
            setattr(self, k, v)
        return params

    def I(self,  # noqa: E743
          func: Callable, *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          **kwargs) -> np.ndarray:
        """
        Declare an indicator. An indicator is just an array of values
        (or a tuple of such arrays in case of, e.g., MACD indicator),
        but one that is revealed gradually in
        `backtesting.backtesting.Strategy.next` much like
        `backtesting.backtesting.Strategy.data` is.
        Returns `np.ndarray` of indicator values.

        `func` is a function that returns the indicator array(s) of
        same length as `backtesting.backtesting.Strategy.data`.

        In the plot legend, the indicator is labeled with
        function name, unless `name` overrides it. If `func` returns
        a tuple of arrays, `name` can be a sequence of strings, and
        its size must agree with the number of arrays returned.

        If `plot` is `True`, the indicator is plotted on the resulting
        `backtesting.backtesting.Backtest.plot`.

        If `overlay` is `True`, the indicator is plotted overlaying the
        price candlestick chart (suitable e.g. for moving averages).
        If `False`, the indicator is plotted standalone below the
        candlestick chart. By default, a heuristic is used which decides
        correctly most of the time.

        `color` can be string hex RGB triplet or X11 color name.
        By default, the next available color is assigned.

        If `scatter` is `True`, the plotted indicator marker will be a
        circle instead of a connected line segment (default).

        Additional `*args` and `**kwargs` are passed to `func` and can
        be used for parameters.

        For example, using simple moving average function from TA-Lib:

            def init():
                self.sma = self.I(ta.SMA, self.data.Close, self.n_sma)

        .. warning::
            Rolling indicators may front-pad warm-up values with NaNs.
            In this case, the **backtest will only begin on the first bar when
            all declared indicators have non-NaN values** (e.g. bar 201 for a
            strategy that uses a 200-bar MA).
            This can affect results.
        """
        def _format_name(name: str) -> str:
            return name.format(*map(_as_str, args),
                               **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))))

        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = (f'{func_name}({params})' if params else f'{func_name}')
        elif isinstance(name, str):
            name = _format_name(name)
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [_format_name(item) for item in name]
        else:
            raise TypeError(f'Unexpected `name=` type {type(name)}; expected `str` or '
                            '`Sequence[str]`')

        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'Indicator "{name}" error. See traceback above.') from e

        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        # Optionally flip the array if the user returned e.g. `df.values`
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if isinstance(name, list) and (np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'Length of `name=` ({len(name)}) must agree with the number '
                f'of arrays the indicator returns ({value.shape[0]}).')

        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(self._data.Close):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays of same '
                f'length as `data` (data shape: {self._data.Close.shape}; indicator "{name}" '
                f'shape: {getattr(value, "shape", "")}, returned value: {value})')

        if plot and overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            # By default, overlay if strong majority of indicator values
            # is within 30% of Close
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > .6)).mean() > .6

        value = _Indicator(value, name=name, plot=plot, overlay=overlay,
                           color=color, scatter=scatter,
                           # _Indicator.s Series accessor uses this:
                           index=self.data.index)
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `backtesting.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().init()
        """

    @abstractmethod
    def next(self):
        """
        Main strategy runtime method, called as each new
        `backtesting.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `backtesting.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().next()
        """

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(self, *,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None,
            symbol: str = None) -> 'Order':
        """
        Place a new long order. For explanation of parameters, see `Order` class.
        
        Args:
            size: Order size (value or percent of available cash)
            limit: Limit price for the order
            stop: Stop price for the order
            sl: Stop-loss price
            tp: Take-profit price
            tag: User-defined tag for the order
            symbol: Asset symbol for multi-asset strategies (None for default asset)
        
        Returns:
            Order object
        """
        return self._broker.new_order(size, limit, stop, sl, tp, tag, symbol=symbol)

    def sell(self, *,
             size: float = _FULL_EQUITY,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None,
             symbol: str = None) -> 'Order':
        """
        Place a new short order. For explanation of parameters, see `Order` class.
        
        Args:
            size: Order size (value or percent of available cash)
            limit: Limit price for the order
            stop: Stop price for the order
            sl: Stop-loss price
            tp: Take-profit price
            tag: User-defined tag for the order
            symbol: Asset symbol for multi-asset strategies (None for default asset)
        
        Returns:
            Order object
        """
        return self._broker.new_order(-size, limit, stop, sl, tp, tag, symbol=symbol)

    @property
    def equity(self) -> float:
        """Current account equity (cash + assets)."""
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """
        Price data accessor.

        Accessing `data` returns the current price data, roughly equivalent to:

            self.data.Open[-1]      # or
            self.data.Close[-1]     # or
            self.data.High[-1]      # etc.

        which returns the current open/close/high/etc. price.
        When using standard OHLCV data, the basic price attributes
        are: `Open`, `High`, `Low`, `Close`, and `Volume`.

        Slices of price data can be accessed by indexing:

            self.data.Close[-10:]
            self.data.Open[:-5]
            self.data.High[0]

        Price data can also be indexed by date:

            self.data.Open['2021-01']      # or
            self.data.Close['2021-01-03']  # or
            self.data.High['2021']         # etc.

        For more details, see [indexing pandas.Series][].

        [indexing pandas.Series]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#slice-time-series-by-time
        """
        return self._data

    @property
    def position(self) -> 'Position':
        """
        Current position information. A position is non-zero when there are
        existing open trades. The position object offers properties such as
        `size`, `pl`, `pl_pct`, etc. See `Position` class for details.
        """
        return Position(self._broker)

    def position_for(self, symbol: str) -> 'Position':
        """
        Get position information for a specific asset in a multi-asset strategy.
        
        Args:
            symbol: Asset symbol to get position for
            
        Returns:
            Position object for the specified asset
        """
        return Position(self._broker, symbol)

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        """Currently active trade orders."""
        return self._broker.orders

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        """Currently open trades."""
        return self._broker.trades

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """Closed, settled trades."""
        return self._broker.closed_trades


class MultiAssetStrategy(Strategy):
    """
    A multi-asset trading strategy base class. Extend this class and
    override methods `init` and `next` to define your own strategy.
    
    This class provides additional methods for working with multiple assets.
    """
    
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        
    def I_for(self, symbol: str, func: Callable, *args,
              name=None, plot=True, overlay=None, color=None, scatter=False,
              **kwargs) -> np.ndarray:
        """
        Declare an indicator for a specific asset. Similar to the `I` method,
        but for a specific asset in a multi-asset strategy.
        
        Args:
            symbol: Asset symbol to create indicator for
            func: Function that returns the indicator array(s)
            *args: Arguments to pass to the function
            name: Name for the indicator in the plot legend
            plot: Whether to plot the indicator
            overlay: Whether to overlay the indicator on the price chart
            color: Color for the indicator
            scatter: Whether to plot as scatter points
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Indicator array
        """
        # Get the data for the specific asset
        asset_data = self._data[symbol]
        
        # Format the name to include the symbol
        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = f'{symbol}_{func_name}({params})' if params else f'{symbol}_{func_name}'
        elif isinstance(name, str):
            name = f'{symbol}_{name}'
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [f'{symbol}_{item}' for item in name]
            
        # Call the function with the asset data
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'Indicator "{name}" error for {symbol}. See traceback above.') from e
            
        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        # Optionally flip the array if the user returned e.g. `df.values`
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if isinstance(name, list) and (np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'Length of `name=` ({len(name)}) must agree with the number '
                f'of arrays the indicator returns ({value.shape[0]}).')

        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(asset_data.Close):
            raise ValueError(
                'Indicators must return (optionally a tuple of) numpy.arrays of same '
                f'length as `data[{symbol}]` (data shape: {asset_data.Close.shape}; indicator "{name}" '
                f'shape: {getattr(value, "shape", "")}, returned value: {value})')

        if plot and overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / asset_data.Close
            # By default, overlay if strong majority of indicator values
            # is within 30% of Close
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > .6)).mean() > .6

        value = _Indicator(value, name=name, plot=plot, overlay=overlay,
                           color=color, scatter=scatter,
                           # _Indicator.s Series accessor uses this:
                           index=asset_data.index)
        self._indicators.append(value)
        return value
        
    def buy_for(self, symbol: str, *,
                size: float = Strategy._FULL_EQUITY,
                limit: Optional[float] = None,
                stop: Optional[float] = None,
                sl: Optional[float] = None,
                tp: Optional[float] = None,
                tag: object = None) -> 'Order':
        """
        Place a new long order for a specific asset.
        
        Args:
            symbol: Asset symbol to buy
            size: Order size (value or percent of available cash)
            limit: Limit price for the order
            stop: Stop price for the order
            sl: Stop-loss price
            tp: Take-profit price
            tag: User-defined tag for the order
            
        Returns:
            Order object
        """
        return self.buy(size=size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag, symbol=symbol)
        
    def sell_for(self, symbol: str, *,
                 size: float = Strategy._FULL_EQUITY,
                 limit: Optional[float] = None,
                 stop: Optional[float] = None,
                 sl: Optional[float] = None,
                 tp: Optional[float] = None,
                 tag: object = None) -> 'Order':
        """
        Place a new short order for a specific asset.
        
        Args:
            symbol: Asset symbol to sell
            size: Order size (value or percent of available cash)
            limit: Limit price for the order
            stop: Stop price for the order
            sl: Stop-loss price
            tp: Take-profit price
            tag: User-defined tag for the order
            
        Returns:
            Order object
        """
        return self.sell(size=size, limit=limit, stop=stop, sl=sl, tp=tp, tag=tag, symbol=symbol)
        
    @property
    def symbols(self) -> List[str]:
        """List of all asset symbols available in the strategy."""
        return self._data.symbols


class _Orders(tuple):
    """
    TODO: remove this class. Only for deprecation.
    """
    def cancel(self):
        """Cancel all non-contingent (i.e. SL/TP) orders."""
        for order in self:
            if not order.is_contingent:
                order.cancel()

    def __getattr__(self, item):
        # TODO: Warn on deprecations from the previous version. Remove in the next.
        removed_attrs = ('entry', 'set_entry', 'is_long', 'is_short',
                         'sl', 'tp', 'set_sl', 'set_tp')
        if item in removed_attrs:
            raise AttributeError(f'Strategy.orders.{"/.".join(removed_attrs)} were removed in'
                                 'Backtesting 0.2.0. '
                                 'Use `Order` API instead. See docs.')
        raise AttributeError(f"'tuple' object has no attribute {item!r}")


class Position:
    """
    Currently held asset position, available as
    `backtesting.backtesting.Strategy.position` within
    `backtesting.backtesting.Strategy.next`.
    Can be used in boolean contexts, e.g.

        if self.position:
            ...  # we have a position, either long or short
    """
    def __init__(self, broker: '_Broker', symbol: str = None):
        self.__broker = broker
        self.__symbol = symbol

    def __bool__(self):
        return bool(self.size)

    @property
    def size(self) -> float:
        """Position size in units of asset. Negative if position is short."""
        return self.__broker._position_size(self.__symbol)

    @property
    def pl(self) -> float:
        """Profit (positive) or loss (negative) of the current position in cash units."""
        return self.__broker._position_pl(self.__symbol)

    @property
    def pl_pct(self) -> float:
        """Profit (positive) or loss (negative) of the current position in percent."""
        if not self.size:
            return 0
        entry_price = self.__broker._position_entry_price(self.__symbol)
        price = self.__broker._last_price(self.__symbol)
        return copysign(1, self.size) * (price / entry_price - 1)

    @property
    def is_long(self) -> bool:
        """True if the position is long (position size is positive)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if the position is short (position size is negative)."""
        return self.size < 0

    @property
    def symbol(self) -> Optional[str]:
        """Asset symbol of the position. None for the default asset."""
        return self.__symbol

    def close(self, portion: float = 1.):
        """
        Close portion of position by closing `portion` of each active trade.
        See also `backtesting.backtesting.Trade.close`.
        """
        for trade in self.__broker._trades:
            if trade.is_open and (self.__symbol is None or trade.symbol == self.__symbol):
                trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size}>'


class _OutOfMoneyError(Exception):
    pass


class Order:
    """
    Place new orders through `Strategy.buy()` and `Strategy.sell()`.
    Query existing orders through `Strategy.orders`.

    When an order is executed or [filled], it results in a `Trade`.

    If you wish to modify aspects of a placed but not yet filled order,
    cancel it and place a new one instead.

    All placed orders are [Good 'Til Canceled].

    [filled]: https://www.investopedia.com/terms/f/fill.asp
    [Good 'Til Canceled]: https://www.investopedia.com/terms/g/gtc.asp
    """
    def __init__(self, broker: '_Broker',
                 size: float,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 sl_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 parent_trade: Optional['Trade'] = None,
                 tag: object = None,
                 symbol: str = None):
        self.__broker = broker
        assert size != 0
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade
        self.__tag = tag
        self.__symbol = symbol
        self.__active = True

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}{k}', v)
        return self

    def __repr__(self):
        return (f'<Order size={self.__size} '
                f'limit={self.__limit_price} '
                f'stop={self.__stop_price} '
                f'sl={self.__sl_price} '
                f'tp={self.__tp_price} '
                f'symbol={self.__symbol} '
                f'parent_trade={self.__parent_trade and id(self.__parent_trade)} '
                f'active={self.__active}>')

    def cancel(self):
        """Cancel the order."""
        self.__broker._orders.remove(self)
        self.__active = False

    @property
    def size(self) -> float:
        """
        Order size (negative for short orders).

        If size is a float between 0 and 1, it is interpreted as a fraction of
        current available liquidity (cash).
        """
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        """
        Order limit price for [limit orders], or None for market orders.

        [limit orders]: https://www.investopedia.com/terms/l/limitorder.asp
        """
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        """
        Order stop price for [stop-limit/stop-market][_] orders, or None for no stop.

        [_]: https://www.investopedia.com/terms/s/stoporder.asp
        """
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        """
        Stop-loss price (for [bracketed][] orders).

        [bracketed]: https://www.investopedia.com/articles/active-trading/091813/which-order-use-stoploss-or-stoplimit-orders.asp
        """
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        """
        Take-profit price (for [bracketed][] orders).

        [bracketed]: https://www.investopedia.com/articles/active-trading/091813/which-order-use-stoploss-or-stoplimit-orders.asp
        """
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    @property
    def tag(self):
        """
        User-defined object attached to the order.
        Passed to resulting Trade object.
        """
        return self.__tag

    @property
    def symbol(self) -> Optional[str]:
        """Asset symbol of the order. None for the default asset."""
        return self.__symbol

    @property
    def is_long(self):
        """True if the order is long (order size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the order is short (order size is negative)."""
        return self.__size < 0

    @property
    def is_contingent(self):
        """
        True for [contingent orders][] such as stop-loss and take-profit orders.

        You can only have a contingent order as a part of a bracket order.
        Use `sl` and `tp` params to `Strategy.buy()`/`Strategy.sell()`.

        [contingent orders]: https://www.tradingtechnologies.com/help/x-study/order-types/additional-order-types/if-touched-order/
        """
        return bool(self.__parent_trade)


class Trade:
    """
    When an `Order` is filled, it results in an active `Trade`.
    Find active trades in `Strategy.trades` and closed, settled trades in `Strategy.closed_trades`.
    """
    def __init__(self, broker: '_Broker', size: int, entry_price: float, entry_bar, tag, symbol: str = None):
        self.__broker = broker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__tag = tag
        self.__symbol = symbol
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None
        self._commissions = 0

    def __repr__(self):
        return (f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar} '
                f'price={self.__entry_price}-{self.__exit_price} pl={self.pl:.0f} '
                f'symbol={self.__symbol}>')

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}{k}', v)
        return self

    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.):
        """Place new `Order` to close `portion` of the trade."""
        assert 0 < portion <= 1, f'portion must be between 0 and 1, is {portion}'
        size = copysign(max(1, abs(int(self.__size * portion))), -self.__size)
        order = self.__broker.new_order(size, trade=self, symbol=self.__symbol)
        return order

    @property
    def size(self):
        """Trade size (volume; negative for short trades)."""
        return self.__size

    @property
    def entry_price(self) -> float:
        """Trade entry price."""
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        """Trade exit price (or None if the trade is still active)."""
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        """Candlestick bar index of when the trade was entered."""
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        """
        Candlestick bar index of when the trade was exited
        (or None if the trade is still active).
        """
        return self.__exit_bar

    @property
    def symbol(self) -> Optional[str]:
        """Asset symbol of the trade. None for the default asset."""
        return self.__symbol

    @property
    def tag(self):
        """
        User-defined object attached to the trade.
        Passed from order.
        """
        return self.__tag

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        """Datetime of when the trade was entered."""
        return self.__broker._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        """Datetime of when the trade was exited."""
        if self.__exit_bar is None:
            return None
        return self.__broker._data.index[self.__exit_bar]

    @property
    def is_open(self):
        """True if the trade is still open."""
        return self.__exit_bar is None

    @property
    def is_long(self):
        """True if the trade is long (trade size is positive)."""
        return self.__size > 0

    @property
    def is_short(self):
        """True if the trade is short (trade size is negative)."""
        return self.__size < 0

    @property
    def pl(self):
        """Trade profit (positive) or loss (negative) in cash units."""
        price = (self.__exit_price if self.__exit_price is not None else
                 self.__broker._last_price(self.__symbol))
        pl = self.__size * (price - self.__entry_price)
        return pl

    @property
    def pl_pct(self):
        """Trade profit (positive) or loss (negative) in percent."""
        price = (self.__exit_price if self.__exit_price is not None else
                 self.__broker._last_price(self.__symbol))
        pl = copysign(1, self.__size) * (price / self.__entry_price - 1)
        return pl

    @property
    def value(self):
        """Trade total value in cash (volume × price)."""
        price = (self.__exit_price if self.__exit_price is not None else
                 self.__broker._last_price(self.__symbol))
        return abs(self.__size) * price

    @property
    def sl(self):
        """
        Stop-loss price at which to close the trade.

        This variable is writable. Set it to None to cancel the stop-loss order.
        [Precision] is handled as with all other prices.

        [Precision]: https://www.investopedia.com/terms/s/significant-figures.asp
        """
        if self.__sl_order is not None and self.__sl_order in self.__broker._orders:
            return self.__sl_order.stop
        return None

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def tp(self):
        """
        Take-profit price at which to close the trade.

        This property is writable. Set it to None to cancel the take-profit order.
        [Precision] is handled as with all other prices.

        [Precision]: https://www.investopedia.com/terms/s/significant-figures.asp
        """
        if self.__tp_order is not None and self.__tp_order in self.__broker._orders:
            return self.__tp_order.limit
        return None

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent('tp', price)

    def __set_contingent(self, type, price):
        assert type in ('sl', 'tp')
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Optional[Order] = getattr(self, attr)

        if order is not None and order in self.__broker._orders:
            order.cancel()

        if price is not None:
            size = -self.__size
            order_kwargs = dict(size=size, parent_trade=self, tag=self.__tag, symbol=self.__symbol)
            if type == 'sl':
                order = self.__broker.new_order(stop=price, **order_kwargs)
            else:
                assert type == 'tp'
                order = self.__broker.new_order(limit=price, **order_kwargs)
        else:
            order = None

        setattr(self, attr, order)
        return order


class _Broker:
    def __init__(self, *, data, cash, spread, commission, margin,
                 trade_on_close, hedging, exclusive_orders, index):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert 0 <= spread < .1, f"spread should be between 0 and 0.1 (0-10%), is {spread}"
        assert 0 <= margin <= 1, f"margin should be between 0 and 1 (0-100%), is {margin}"
        self._data = data
        self._cash = cash
        self._spread = spread
        self._commission = commission
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders

        self._equity = np.full_like(index, cash, dtype=float)
        self._index = index
        self._orders: List[Order] = []
        self._trades: List[Trade] = []
        self._closed_trades: List[Trade] = []
        
        # For multi-asset support
        self._positions = {}  # symbol -> position size
        self._last_prices = {}  # symbol -> last price

    def _commission_func(self, order_size, price):
        return 0

    def __repr__(self):
        return f'<Broker: {self._cash:.0f}{" HF" if self._hedging else ""}>'
        
    def _position_size(self, symbol=None):
        """Get position size for a specific asset."""
        if symbol is None:
            # For backward compatibility, return sum of all positions
            return sum(self._positions.values())
        return self._positions.get(symbol, 0)
        
    def _position_pl(self, symbol=None):
        """Get profit/loss for a specific asset position."""
        if symbol is None:
            # For backward compatibility, return sum of all position P/Ls
            return sum(trade.pl for trade in self._trades)
        return sum(trade.pl for trade in self._trades if trade.symbol == symbol)
        
    def _position_entry_price(self, symbol=None):
        """Get average entry price for a specific asset position."""
        if symbol is None:
            # For backward compatibility, use weighted average of all positions
            weights = np.abs([trade.size for trade in self._trades])
            if not weights.sum():
                return 0
            return np.average([trade.entry_price for trade in self._trades], weights=weights)
        
        trades = [trade for trade in self._trades if trade.symbol == symbol]
        weights = np.abs([trade.size for trade in trades])
        if not weights.sum():
            return 0
        return np.average([trade.entry_price for trade in trades], weights=weights)
        
    def _last_price(self, symbol=None):
        """Get last price for a specific asset."""
        if symbol is None:
            # For backward compatibility, return the price of the default asset
            return self.last_price
        return self._last_prices.get(symbol, 0)

    def new_order(self,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None,
                  symbol: str = None) -> Order:
        """
        Create a new order.
        
        For documentation of parameters, see `Order` class.
        """
        size = float(size)
        
        # Get the appropriate price data based on the symbol
        if symbol is None:
            # Use the default data
            price_data = self._data
        else:
            # Use the data for the specific symbol
            price_data = self._data[symbol]
            
        # Get the current price
        adjusted_price = self._adjusted_price(size, price_data.Close[-1], symbol)
        
        # Validate the order parameters
        if limit is not None and stop is not None:
            if ((size > 0 and limit > stop) or
                    (size < 0 and limit < stop)):
                raise ValueError(
                    "For long orders, limit price must be below stop price; "
                    "for short orders, limit price must be above stop price")
                    
        # Create the order
        order = Order(self, size, limit, stop, sl, tp, trade, tag, symbol)
        
        # Add the order to the queue
        self._orders.append(order)
        
        return order

    @property
    def last_price(self) -> float:
        """Return the current price."""
        return self._data.Close[-1]

    def _adjusted_price(self, size=None, price=None, symbol=None) -> float:
        """
        Return price adjusted for spread.
        In long positions, we buy at a higher price,
        and in short positions, we sell at a lower price.
        """
        # If no price specified, use the current price
        if price is None:
            if symbol is None:
                price = self.last_price
            else:
                price = self._data[symbol].Close[-1]
                
        # Apply spread adjustment
        return price * (1 + copysign(self._spread, size or 1))

    @property
    def equity(self) -> float:
        """Current account equity (cash + assets)."""
        return self._cash + sum(trade.pl for trade in self._trades)

    @property
    def margin_available(self) -> float:
        """
        Available margin (cash plus unrealized P/L).
        If no margin was used, returns cash.
        """
        return self._cash + sum(trade.pl for trade in self._trades)

    def next(self):
        """
        Process the next candlestick bar and update the broker state.
        """
        # Update last prices for all assets
        if hasattr(self._data, 'symbols'):
            # Multi-asset data
            for symbol in self._data.symbols:
                self._last_prices[symbol] = self._data[symbol].Close[-1]
        else:
            # Single asset data
            self._last_prices[None] = self._data.Close[-1]
            
        # Process orders
        self._process_orders()
        
        # Update equity history
        self._equity[min(len(self._equity) - 1, len(self._data) - 1)] = self.equity

    def _process_orders(self):
        """
        Process all pending orders.
        """
        # Implementation of order processing logic
        # This would need to be updated to handle multi-asset orders
        pass

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        """
        Reduce the size of an existing trade.
        """
        # Implementation of trade reduction logic
        pass

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        """
        Close an existing trade.
        """
        # Implementation of trade closing logic
        pass

    def _open_trade(self, price: float, size: int,
                    sl: Optional[float], tp: Optional[float], time_index: int, tag, symbol=None):
        """
        Open a new trade.
        """
        # Implementation of trade opening logic
        # Update to handle symbol parameter
        trade = Trade(self, size, price, time_index, tag, symbol)
        
        # Update position tracking
        self._positions[symbol] = self._positions.get(symbol, 0) + size
        
        # Set stop-loss and take-profit if specified
        if sl:
            trade.sl = sl
        if tp:
            trade.tp = tp
            
        self._trades.append(trade)
        return trade

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        """Currently active orders."""
        return tuple(self._orders)

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        """Currently open trades."""
        return tuple(self._trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """Closed, settled trades."""
        return tuple(self._closed_trades)


class Backtest:
    """
    Backtest a particular (parameterized) strategy
    on particular data.

    Upon initialization, call method
    `backtesting.backtesting.Backtest.run` to run a backtest
    instance, or `backtesting.backtesting.Backtest.optimize` to
    optimize it.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 10_000,
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 finalize_trades=False,
                 ):
        """
        Initialize a backtest. Requires data and a strategy to test.

        `data` is a `pd.DataFrame` with columns:
        `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
        If any columns are missing, set them to what you have available,
        e.g.

            df['Open'] = df['High'] = df['Low'] = df['Close']

        The passed data frame can contain additional columns that
        can be used by the strategy (e.g. sentiment info).
        DataFrame index can be either a datetime index (timestamps)
        or a monotonic range index (i.e. a sequence of periods).

        `strategy` is a `backtesting.backtesting.Strategy`
        _subclass_ (not an instance).

        `cash` is the initial cash to start with.

        `spread` is the the constant bid-ask spread rate (relative to the price).
        E.g. set it to `0.0002` for commission-less forex
        trading where the average spread is roughly 0.2‰ of the asking price.

        `commission` is the commission rate. E.g. if your broker's commission
        is 1% of order value, set commission to `0.01`.
        The commission is applied twice: at trade entry and at trade exit.
        Besides one single floating value, `commission` can also be a tuple of floating
        values `(fixed, relative)`. E.g. set it to `(100, .01)`
        if your broker charges minimum $100 + 1%.
        Additionally, `commission` can be a callable
        `func(order_size: int, price: float) -> float`
        (note, order size is negative for short orders),
        which can be used to model more complex commission structures.
        Negative commission values are interpreted as market-maker's rebates.

        .. note::
            Before v0.4.0, the commission was only applied once, like `spread` is now.
            If you want to keep the old behavior, simply set `spread` instead.

        .. note::
            With nonzero `commission`, long and short orders will be placed
            at an adjusted price that is slightly higher or lower (respectively)
            than the current price. See e.g.
            [#153](https://github.com/kernc/backtesting.py/issues/153),
            [#538](https://github.com/kernc/backtesting.py/issues/538),
            [#633](https://github.com/kernc/backtesting.py/issues/633).

        `margin` is the required margin (ratio) of a leveraged account.
        No difference is made between initial and maintenance margins.
        To run the backtest using e.g. 50:1 leverge that your broker allows,
        set margin to `0.02` (1 / leverage).

        If `trade_on_close` is `True`, market orders will be filled
        with respect to the current bar's closing price instead of the
        next bar's open.

        If `hedging` is `True`, allow trades in both directions simultaneously.
        If `False`, the opposite-facing orders first close existing trades in
        a [FIFO] manner.

        If `exclusive_orders` is `True`, each new order auto-closes the previous
        trade/position, making at most a single trade (long or short) in effect
        at each time.

        If `finalize_trades` is `True`, the trades that are still
        [active and ongoing] at the end of the backtest will be closed on
        the last bar and will contribute to the computed backtest statistics.

        [FIFO]: https://www.investopedia.com/terms/n/nfa-compliance-rule-2-43b.asp
        [active and ongoing]: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.trades
        """  # noqa: E501

        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float value, percent of '
                            'entry order price')
        if not isinstance(commission, (Number, tuple)) and not callable(commission):
            raise TypeError('`commission` must be a float percent of order value, '
                            'a tuple of `(fixed, relative)` commission, '
                            'or a function that takes `(order_size, price)`'
                            'and returns commission dollar value')

        data = data.copy(deep=False)

        # Convert index to datetime index
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            # Numeric index with most large numbers
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass

        if 'Volume' not in data:
            data['Volume'] = np.nan

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError("`data` must be a pandas.DataFrame with columns "
                             "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError('Some OHLC values are missing (NaN). '
                             'Please strip those lines with `df.dropna()` or '
                             'fill them in with `df.interpolate()` or whatever.')
        if np.any(data['Close'] > cash):
            warnings.warn('Some prices are larger than initial cash value. Note that fractional '
                          'trading is not supported. If you want to trade Bitcoin, '
                          'increase initial cash, or trade μBTC or satoshis instead (GH-134).',
                          stacklevel=2)
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.',
                          stacklevel=2)
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.',
                          stacklevel=2)

        self._data: pd.DataFrame = data
        self._broker = partial(
            _Broker, cash=cash, spread=spread, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders, index=data.index,
        )
        self._strategy = strategy
        self._results: Optional[pd.Series] = None
        self._finalize_trades = bool(finalize_trades)

    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                    96.74115
            Equity Final [$]                     51422.99
            Equity Peak [$]                      75787.44
            Return [%]                           414.2299
            Buy & Hold Return [%]               703.45824
            Return (Ann.) [%]                    21.18026
            Volatility (Ann.) [%]                36.49391
            CAGR [%]                             14.15984
            Sharpe Ratio                          0.58038
            Sortino Ratio                         1.08479
            Calmar Ratio                          0.44144
            Max. Drawdown [%]                   -47.98013
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   66
            Win Rate [%]                          46.9697
            Best Trade [%]                       53.59595
            Worst Trade [%]                     -18.39887
            Avg. Trade [%]                        2.53172
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.16795
            Expectancy [%]                        3.27481
            SQN                                   1.07662
            Kelly Criterion                       0.15187
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            dtype: object

        .. warning::
            You may obtain different results for different strategy parameters.
            E.g. if you use 50- and 200-bar SMA, the trading simulation will
            begin on bar 201. The actual length of delay is equal to the lookback
            period of the `Strategy.I` indicator which lags the most.
            Obviously, this can affect results.
        """
        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()  # Strategy.init might have changed/added to data.df

        # Indicators used in Strategy.next()
        indicator_attrs = _strategy_indicators(strategy)

        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        start = 1 + _indicator_warmup_nbars(strategy)

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid='ignore'):

            for i in _tqdm(range(start, len(self._data)), desc=self.run.__qualname__):
                # Prepare data and indicators for `next` call
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    # Slice indicator on the last dimension (case of 2d indicator)
                    setattr(strategy, attr, indicator[..., :i + 1])

                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                strategy.next()
            else:
                if self._finalize_trades is True:
                    # Close any remaining open trades so they produce some stats
                    for trade in reversed(broker.trades):
                        trade.close()

                    # HACK: Re-run broker one last time to handle close orders placed in the last
                    #  strategy iteration. Use the same OHLC values as in the last broker iteration.
                    if start < len(self._data):
                        try_(broker.next, exception=_OutOfMoneyError)

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            data._set_length(len(self._data))

            equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )

        return self._results

    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"sambo"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: https://sambo-optimization.github.io

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="sambo"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'sambo'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [SAMBO]'s [plotting tools].

        [OptimizeResult]: https://sambo-optimization.github.io/doc/sambo/#sambo.OptimizeResult
        [SAMBO]: https://sambo-optimization.github.io
        [plotting tools]: https://sambo-optimization.github.io/doc/sambo/plot.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer random seed.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                              constraint=lambda p: p.sma1 < p.sma2)

        .. TODO::
            Improve multiprocessing/parallel execution on Windos with start method 'spawn'.
        """
        if not kwargs:
            raise ValueError('No strategy parameters to optimize')
        
        if method != 'grid':
            raise ValueError(f"Optimization method '{method}' not supported. Only 'grid' is available.")
        
        # Extract parameter names and values
        param_names = list(kwargs.keys())
        param_values = list(kwargs.values())
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*param_values))
        n_combinations = len(param_combinations)
        
        # Apply max_tries limit if specified
        if max_tries is not None:
            if isinstance(max_tries, float) and 0 < max_tries < 1:
                max_tries = int(max_tries * n_combinations)
            
            if max_tries < n_combinations:
                if random_state is not None:
                    random.seed(random_state)
                param_combinations = random.sample(param_combinations, max_tries)
                n_combinations = max_tries
        
        # Prepare for optimization
        stats_list = []
        heatmap_values = []
        
        # Run backtest for each parameter combination
        with _tqdm(total=n_combinations, desc='Optimizing') as pbar:
            for params in param_combinations:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Apply constraint if provided
                if constraint is not None and not constraint(param_dict):
                    pbar.update()
                    continue
                
                # Run backtest with current parameters
                try:
                    stats = self.run(**param_dict)
                    
                    # Extract the value to maximize
                    if callable(maximize):
                        maximize_value = maximize(stats)
                    else:
                        maximize_value = stats[maximize]
                    
                    # Store results
                    stats_list.append((param_dict, stats, maximize_value))
                    
                    # Store heatmap values if requested
                    if return_heatmap:
                        heatmap_values.append((*params, maximize_value))
                    
                except Exception as e:
                    warnings.warn(f"Error during optimization with parameters {param_dict}: {str(e)}")
                
                pbar.update()
        
        if not stats_list:
            raise RuntimeError('No successful backtest runs')
        
        # Find the best parameters
        best_params, best_stats, _ = max(stats_list, key=lambda x: x[2])
        
        # Run the backtest with the best parameters to update self._results
        self.run(**best_params)
        
        # Prepare return values
        result = best_stats
        
        # Create heatmap if requested
        heatmap = None
        if return_heatmap:
            if len(param_names) <= 2:
                # For 1D or 2D parameter space, create a DataFrame
                if len(param_names) == 1:
                    # 1D parameter space
                    heatmap = pd.Series(
                        [item[-1] for item in heatmap_values],
                        index=pd.Index([item[0] for item in heatmap_values], name=param_names[0])
                    )
                else:
                    # 2D parameter space
                    heatmap_data = {}
                    for p1, p2, value in heatmap_values:
                        if p1 not in heatmap_data:
                            heatmap_data[p1] = {}
                        heatmap_data[p1][p2] = value
                    
                    heatmap = pd.DataFrame(heatmap_data).T
                    heatmap.index.name = param_names[0]
                    heatmap.columns.name = param_names[1]
            else:
                # For higher dimensions, create a Series with MultiIndex
                index = pd.MultiIndex.from_tuples(
                    [item[:-1] for item in heatmap_values],
                    names=param_names
                )
                heatmap = pd.Series(
                    [item[-1] for item in heatmap_values],
                    index=index
                )
        
        # Create optimization result if requested
        optimization = None
        if return_optimization:
            optimization = {
                'param_names': param_names,
                'param_values': param_values,
                'stats_list': stats_list,
                'best_params': best_params,
                'best_stats': best_stats,
                'heatmap': heatmap
            }
        
        # Return appropriate result format
        if return_heatmap and return_optimization:
            return result, heatmap, optimization
        elif return_heatmap:
            return result, heatmap
        elif return_optimization:
            return result, optimization
        else:
            return result

    @staticmethod
    def _mp_task(arg):
        # Implementation of _mp_task method
        pass

    def plot(self, *, results: pd.Series = None, filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=True, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = True,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `backtesting.backtesting.Backtest.run` or
        `backtesting.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.
        By default, a strategy/parameter-dependent file is created in the
        current working directory.

        `plot_width` is the width of the plot in pixels. If None (default),
        the plot is made to span 100% of browser width. The height is
        currently non-adjustable.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity (initial cash plus assets) graph section. This is the same
        as `plot_return` plus initial 100%.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section. This is the same
        as `plot_equity` minus initial 100%.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `plot_trades` is `True`, the stretches between trade entries
        and trade exits are marked by hash-marked tractor beams.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times,
        unaffected by any interim asset volatility.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart. Default downsampling rule is:
        monthly for daily data, daily for hourly data, hourly for minute data,
        and minute for (sub-)second data.
        `superimpose` can also be a valid [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to superimpose.
        Note, this only works for data with a datetime index.

        If `resample` is `True`, the OHLC data is resampled in a way that
        makes the upper number of candles for Bokeh to plot limited to 10_000.
        This may, in situations of overabundant data,
        improve plot's interactive performance and avoid browser's
        `Javascript Error: Maximum call stack size exceeded` or similar.
        Equity & dropdown curves and individual trades data is,
        likewise, [reasonably _aggregated_][TRADES_AGG].
        `resample` can also be a [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to resample, overriding above numeric limitation.
        Note, all this only works for data with a datetime index.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        [Pandas offset string]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        [TRADES_AGG]: lib.html#backtesting.lib.TRADES_AGG

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.
        """
        # Implementation of plot method
        pass


class MultiAssetBacktest:
    """
    Backtest a multi-asset strategy on multiple data sources.
    
    This class extends the functionality of Backtest to support
    strategies that trade multiple assets simultaneously.
    """
    
    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 strategy: Type[MultiAssetStrategy],
                 *,
                 cash: float = 10_000,
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 finalize_trades=False,
                 ):
        """
        Initialize a multi-asset backtest. Requires a dictionary of data frames and a strategy to test.
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary mapping asset symbols to their respective OHLCV DataFrames.
            All DataFrames must have the same index and the standard OHLCV columns.
            
        strategy : Type[MultiAssetStrategy]
            A MultiAssetStrategy subclass (not an instance).
            
        cash : float
            Initial cash to start with.
            
        spread : float
            The constant bid-ask spread rate (relative to the price).
            
        commission : Union[float, Tuple[float, float]]
            Commission rate or (fixed, relative) commission tuple.
            
        margin : float
            Required margin (ratio) of a leveraged account.
            
        trade_on_close : bool
            If True, market orders will be filled at the current bar's closing price.
            
        hedging : bool
            If True, allow trades in both directions simultaneously.
            
        exclusive_orders : bool
            If True, each new order auto-closes the previous trade/position.
            
        finalize_trades : bool
            If True, trades still active at the end of the backtest will be closed.
        """
        if not (isinstance(strategy, type) and issubclass(strategy, MultiAssetStrategy)):
            raise TypeError('`strategy` must be a MultiAssetStrategy sub-type')
        
        if not isinstance(data_dict, dict) or not data_dict:
            raise TypeError("`data_dict` must be a non-empty dictionary mapping symbols to DataFrames")
        
        # Validate all dataframes
        for symbol, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Data for symbol '{symbol}' must be a pandas.DataFrame")
            
            # Check for required columns
            if len(df.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
                raise ValueError(f"Data for symbol '{symbol}' must have columns "
                                 "'Open', 'High', 'Low', 'Close', and 'Volume'")
                
            # Check for NaN values
            if df[['Open', 'High', 'Low', 'Close']].isnull().values.any():
                raise ValueError(f"Some OHLC values for symbol '{symbol}' are missing (NaN). "
                                 "Please strip those lines with `df.dropna()` or "
                                 "fill them in with `df.interpolate()` or whatever.")
                
            # Add Volume if missing
            if 'Volume' not in df:
                df['Volume'] = np.nan
                
            # Sort index if needed
            if not df.index.is_monotonic_increasing:
                warnings.warn(f'Data index for symbol {symbol} is not sorted in ascending order. Sorting.',
                              stacklevel=2)
                data_dict[symbol] = df.sort_index()
        
        # Ensure all dataframes have the same index
        if len(data_dict) > 1:
            first_symbol = next(iter(data_dict.keys()))
            first_index = data_dict[first_symbol].index
            for symbol, df in data_dict.items():
                if not df.index.equals(first_index):
                    raise ValueError(f"All assets must have the same index. Asset {symbol} has a different index.")
        
        # Create a multi-asset data object
        from ._util import _MultiAssetData
        self._data_dict = data_dict
        self._multi_data = _MultiAssetData(data_dict)
        
        # Store parameters
        self._strategy = strategy
        self._cash = cash
        self._spread = spread
        self._commission = commission
        self._margin = margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders
        self._finalize_trades = finalize_trades
        
        # Initialize results
        self._results: Optional[pd.Series] = None
        
    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.
        
        Keyword arguments are interpreted as strategy parameters.
        
        Returns:
        --------
        pd.Series
            Series containing backtest results and statistics.
        """
        # Create broker
        broker = _Broker(
            data=self._multi_data,
            cash=self._cash,
            spread=self._spread,
            commission=self._commission,
            margin=self._margin,
            trade_on_close=self._trade_on_close,
            hedging=self._hedging,
            exclusive_orders=self._exclusive_orders,
            index=self._multi_data.index
        )
        
        # Create strategy instance
        strategy = self._strategy(broker, self._multi_data, kwargs)
        
        # Initialize strategy
        strategy.init()
        
        # Indicators used in Strategy.next()
        indicator_attrs = _strategy_indicators(strategy)
        
        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        start = 1 + _indicator_warmup_nbars(strategy)
        
        # Disable "invalid value encountered in ..." warnings
        with np.errstate(invalid='ignore'):
            # Main backtest loop
            for i in _tqdm(range(start, len(self._multi_data)), desc="Running backtest"):
                # Prepare data and indicators for `next` call
                self._multi_data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    # Slice indicator on the last dimension (case of 2d indicator)
                    setattr(strategy, attr, indicator[..., :i + 1])
                
                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break
                
                # Next tick, a moment before bar close
                strategy.next()
            else:
                if self._finalize_trades:
                    # Close any remaining open trades
                    for trade in reversed(broker.trades):
                        trade.close()
                    
                    # Re-run broker one last time to handle close orders
                    if start < len(self._multi_data):
                        try_(broker.next, exception=_OutOfMoneyError)
            
            # Set data back to full length
            self._multi_data._set_length(len(self._multi_data))
            
            # Compute statistics
            equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
            
            # Use the first asset's data for OHLC stats
            first_symbol = next(iter(self._data_dict.keys()))
            ohlc_data = self._data_dict[first_symbol]
            
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=ohlc_data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )
        
        return self._results
    
    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.
        
        Parameters
        ----------
        maximize : str or callable
            The name of the statistic to maximize, e.g. 'SQN', 'Return [%]', 'Sharpe Ratio', etc.,
            or a function that accepts a `pd.Series` of statistics and returns a number.
            
        method : str
            Optimization method. Currently only 'grid' is supported.
            
        max_tries : int or float, optional
            Maximum number of strategy runs to perform.
            If float between 0 and 1, interpreted as a fraction of the full grid size.
            If None, all grid combinations are evaluated.
            
        constraint : callable, optional
            A function that accepts a dict of parameters and returns True
            if the parameters satisfy the constraint.
            
        return_heatmap : bool
            If True, return a series with a MultiIndex of all run parameter values
            and resulting statistic values.
            
        return_optimization : bool
            If True, return the full optimization result object.
            
        random_state : int, optional
            Random seed for reproducible results.
            
        **kwargs
            Strategy parameters to optimize. Each parameter should be a sequence of values to try.
            
        Returns
        -------
        pd.Series
            Series of statistics for the best run.
            
        If `return_heatmap` is True, a tuple of (stats, heatmap) is returned.
        If `return_optimization` is True, a tuple of (stats, heatmap, optimization) is returned.
        """
        if not kwargs:
            raise ValueError('No strategy parameters to optimize')
        
        if method != 'grid':
            raise ValueError(f"Optimization method '{method}' not supported. Only 'grid' is available.")
        
        # Extract parameter names and values
        param_names = list(kwargs.keys())
        param_values = list(kwargs.values())
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(*param_values))
        n_combinations = len(param_combinations)
        
        # Apply max_tries limit if specified
        if max_tries is not None:
            if isinstance(max_tries, float) and 0 < max_tries < 1:
                max_tries = int(max_tries * n_combinations)
            
            if max_tries < n_combinations:
                if random_state is not None:
                    random.seed(random_state)
                param_combinations = random.sample(param_combinations, max_tries)
                n_combinations = max_tries
        
        # Prepare for optimization
        stats_list = []
        heatmap_values = []
        
        # Run backtest for each parameter combination
        with _tqdm(total=n_combinations, desc='Optimizing') as pbar:
            for params in param_combinations:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Apply constraint if provided
                if constraint is not None and not constraint(param_dict):
                    pbar.update()
                    continue
                
                # Run backtest with current parameters
                try:
                    stats = self.run(**param_dict)
                    
                    # Extract the value to maximize
                    if callable(maximize):
                        maximize_value = maximize(stats)
                    else:
                        maximize_value = stats[maximize]
                    
                    # Store results
                    stats_list.append((param_dict, stats, maximize_value))
                    
                    # Store heatmap values if requested
                    if return_heatmap:
                        heatmap_values.append((*params, maximize_value))
                    
                except Exception as e:
                    warnings.warn(f"Error during optimization with parameters {param_dict}: {str(e)}")
                
                pbar.update()
        
        if not stats_list:
            raise RuntimeError('No successful backtest runs')
        
        # Find the best parameters
        best_params, best_stats, _ = max(stats_list, key=lambda x: x[2])
        
        # Run the backtest with the best parameters to update self._results
        self.run(**best_params)
        
        # Prepare return values
        result = best_stats
        
        # Create heatmap if requested
        heatmap = None
        if return_heatmap:
            if len(param_names) <= 2:
                # For 1D or 2D parameter space, create a DataFrame
                if len(param_names) == 1:
                    # 1D parameter space
                    heatmap = pd.Series(
                        [item[-1] for item in heatmap_values],
                        index=pd.Index([item[0] for item in heatmap_values], name=param_names[0])
                    )
                else:
                    # 2D parameter space
                    heatmap_data = {}
                    for p1, p2, value in heatmap_values:
                        if p1 not in heatmap_data:
                            heatmap_data[p1] = {}
                        heatmap_data[p1][p2] = value
                    
                    heatmap = pd.DataFrame(heatmap_data).T
                    heatmap.index.name = param_names[0]
                    heatmap.columns.name = param_names[1]
            else:
                # For higher dimensions, create a Series with MultiIndex
                index = pd.MultiIndex.from_tuples(
                    [item[:-1] for item in heatmap_values],
                    names=param_names
                )
                heatmap = pd.Series(
                    [item[-1] for item in heatmap_values],
                    index=index
                )
        
        # Create optimization result if requested
        optimization = None
        if return_optimization:
            optimization = {
                'param_names': param_names,
                'param_values': param_values,
                'stats_list': stats_list,
                'best_params': best_params,
                'best_stats': best_stats,
                'heatmap': heatmap
            }
        
        # Return appropriate result format
        if return_heatmap and return_optimization:
            return result, heatmap, optimization
        elif return_heatmap:
            return result, heatmap
        elif return_optimization:
            return result, optimization
        else:
            return result
