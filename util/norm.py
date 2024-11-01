# %%
import pandas as pd
import numpy as np

# 3. 标准化->离散化降频因子
def normalize_factor(factor: pd.Series, window: int = 2000) -> pd.Series:
    _factor = ((factor - factor.rolling(window=window).mean()) / factor.rolling(window=window).std()).clip(-3, 3)
    _factor = discretize_factor(_factor, step=0.25)
    return _factor


def discretize_factor(factor: pd.Series, step: float = 0.25) -> pd.Series:
    min_val, max_val = -3, 3

    # 使用apply函数对每个元素应用discretize_value函数
    def discretize_value(value):
        # 将因子值映射到离散化的范围内
        discretized_value = np.round((value - min_val) / step) * step + min_val
        # 确保结果在指定的范围内
        return min(max(discretized_value, min_val), max_val)

    return factor.apply(discretize_value)