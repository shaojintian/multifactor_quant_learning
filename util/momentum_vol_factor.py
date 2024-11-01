import pandas as pd
import numpy as np
import talib as ta
from norm import normalize_factor

def adaptive_momentum_factor(data: pd.DataFrame, 
                           momentum_window: int = 20,
                           vol_window: int = 100,
                           std_window: int = 2000) -> pd.Series:
    """
    自适应动量因子，结合:
    1. 价格动量
    2. 波动率调整
    3. 成交量确认
    """
    # 1. 计算动量信号
    returns = data['close'].pct_change()
    momentum = returns.rolling(window=momentum_window).mean()
    
    # 2. 波动率调整
    volatility = returns.rolling(window=vol_window).std()
    vol_adjusted_momentum = momentum / volatility
    
    # 3. 成交量确认
    volume_ma = data['volume'].rolling(window=momentum_window).mean()
    volume_ratio = data['volume'] / volume_ma
    
    # 4. RSI作为超买超卖指标
    rsi = ta.RSI(data['close'].values, timeperiod=14)
    rsi_factor = (rsi - 50) / 50  # 归一化到 [-1, 1]
    
    # 5. 组合信号
    factor = vol_adjusted_momentum * np.sign(volume_ratio - 1) * (1 + rsi_factor)
    

    
    # 设置因子名称
    factor.name = 'adaptive_momentum'
    
    return normalize_factor(factor)
