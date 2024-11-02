import pandas as pd
import numpy as np
from util.norm import normalize_factor

def volatility_adjusted_momentum(df, price_col='close', vol_window=20):
    """
    计算波动率调整后的动量因子
    
    Parameters:
    df: DataFrame with OHLCV data
    price_col: 用于计算的价格列名
    vol_window: 波动率计算窗口
    
    Returns:
    Series with factor values
    """
    returns = df[price_col].pct_change()
    volatility = returns.rolling(vol_window).std()
    return normalize_factor(returns / (volatility + 1e-6))

def volume_weighted_momentum(df, volume_window=10):
    """
    计算成交量加权价格动量
    """
    returns = df['close'].pct_change()
    volume_ratio = df['volume'] / df['volume'].rolling(volume_window).mean()
    return normalize_factor(returns * volume_ratio)

def buy_pressure(df, window=10):
    """
    计算买卖压力因子
    """
    return normalize_factor((df['taker buy base asset volume'] / df['volume']).rolling(window).mean())

def price_efficiency(df):
    """
    计算价格波动效率因子
    """
    high_low_range = (df['high'] - df['low']).abs()
    close_change = df['close'].diff().abs()
    return normalize_factor(close_change / (high_low_range + 1e-6))

def price_volume_divergence(df, window=20):
    """
    计算量价背离因子
    """
    price_ma = df['close'].rolling(window).mean()
    volume_ma = df['volume'].rolling(window).mean()
    return normalize_factor((df['close'] / price_ma - df['volume'] / volume_ma))

def volatility_regime(df, short_window=5, long_window=60, lookback=20):
    """
    计算波动率regime因子
    """
    returns = df['close'].pct_change()
    current_vol = returns.rolling(short_window).std()
    historical_vol = returns.rolling(long_window).std()
    
    vol_min = historical_vol.rolling(lookback).min()
    vol_max = historical_vol.rolling(lookback).max()
    
    return normalize_factor((current_vol - vol_min) / (vol_max - vol_min + 1e-6))

def trade_activity(df, window=20):
    """
    计算交易活跃度因子
    """
    return normalize_factor((df['number of trades'] / df['number of trades'].rolling(window).mean() - 1))

def price_strength(df):
    """
    计算价格区间强度因子
    """
    return normalize_factor((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6))

def volume_imbalance(df, window=10):
    """
    计算买卖量不平衡因子
    """
    taker_buy_ratio = df['taker buy quote asset volume'] / df['quote asset volume']
    return normalize_factor(taker_buy_ratio.rolling(window).mean() - 0.5)

def multi_period_momentum(df, lookback_periods=[5, 10, 20]):
    """
    计算多周期动量组合因子
    """
    momentum_signals = []
    for period in lookback_periods:
        mom = (df['close'] / df['close'].shift(period) - 1)
        momentum_signals.append(mom)
    return normalize_factor(sum(momentum_signals) / len(momentum_signals))

