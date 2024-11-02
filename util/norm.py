# %%
import pandas as pd
import numpy as np

# 3. 滚动标准化->离散化降频因子
import pandas as pd

def normalize_factor(factor: pd.Series, window: int = 2000) -> pd.Series:
    """
    Normalize the factor using rolling mean and standard deviation or global mean and std if the sample size is small.

    Parameters:
    -----------
    factor: pd.Series
        The factor values to normalize.
    window: int
        The rolling window size.

    Returns:
    --------
    pd.Series
        The normalized factor values.
    """
    # Check the length of the factor series
    if len(factor) < window:
         # 方法2：基于百分位数的缩放
        upper = factor.quantile(0.9973)  # 3σ对应的概率约为99.73%
        lower = factor.quantile(0.0027)
        _factor = 6 * (factor - lower) / (upper - lower) - 3
        _factor = _factor.clip(-3, 3)
    else:
        # Use rolling mean and std
        _factor = ((factor - factor.rolling(window=window).mean()) / factor.rolling(window=window).std()).clip(-3, 3)

    #print(_factor.describe())
    return _factor.fillna(0)


def discretize_factor(factor: pd.Series, step: float = 0.25) -> pd.Series:
    min_val, max_val = -3, 3

    # 使用apply函数对每个元素应用discretize_value函数
    def discretize_value(value):
        # 将因子值映射到离散化的范围内
        discretized_value = np.round((value - min_val) / step) * step + min_val
        # 确保结果在指定的范围内
        return min(max(discretized_value, min_val), max_val)

    return factor.apply(discretize_value)