import numpy as np
import pandas as pd

def adjust_positions(factors: pd.DataFrame, threshold: float) -> pd.Series:
    """
    根据因子值调整仓位，只有当因子变化超过阈值时才进行调仓。
    
    Parameters:
    -----------
    factors: DataFrame
        因子值数据框
    threshold: float
        调仓阈值
    
    Returns:
    --------
    positions: Series
        调整后的仓位
    """
    # 计算因子变化
    factor_change = factors.diff().fillna(0)
    
    # 根据阈值调整仓位
    positions = np.where(np.abs(factor_change) >= threshold, factors, 0)
    
    return pd.Series(positions, index=factors.index)

# 使用示例
# factors = pd.DataFrame(...)  # 假设这是你的因子数据
# positions = adjust_positions(factors, threshold=0.01)