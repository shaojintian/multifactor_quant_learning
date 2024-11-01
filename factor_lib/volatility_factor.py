import pandas as pd
import numpy as np
from util.norm import normalize_factor


def calc_vol_mean_reversion_factor(prices: pd.Series, 
                                 lookback_period: int = 20,
                                 vol_window: int = 5) -> pd.Series:
    """
    计算波动率均值回归因子
    
    Args:
        prices: pd.DataFrame, 价格数据，index为时间，columns为资产
        lookback_period: int, 回溯期长度，用于计算历史波动率均值
        vol_window: int, 计算当前波动率的窗口期
    
    Returns:
        pd.DataFrame: 波动率均值回归因子值
    """
    # 计算收益率
    returns = prices.pct_change()
    
    # 计算历史波动率均值
    historical_vol = returns.rolling(window=lookback_period).std()
    historical_vol_mean = historical_vol.rolling(window=lookback_period).mean()
    
    # 计算当前波动率
    current_vol = returns.rolling(window=vol_window).std()
    
    # 计算波动率差异
    vol_diff = (current_vol - historical_vol_mean) / historical_vol_mean
    
    # 因子值取负号，当前波动率高于历史均值时看跌
    factor = -vol_diff
    factor.name = 'vol_mean_reversion_factor'
    
    #print(factor[20:])
    return normalize_factor(factor)