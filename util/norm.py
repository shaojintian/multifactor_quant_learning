# %%
import pandas as pd
import numpy as np

# 3. 滚动标准化->离散化降频因子
import pandas as pd

def normalize_factor(factor: pd.Series, window: int = 2000) -> pd.Series:
    if len(factor) < window:
        upper = factor.quantile(0.9973)
        lower = factor.quantile(0.0027)
        _factor = 6 * (factor - lower) / (upper - lower) - 3
        _factor = _factor.clip(-3, 3)
    else:
        mean_ewm = factor.ewm(span=window, adjust=False).mean()
        std_ewm = factor.ewm(span=window, adjust=False).std()
        _factor = ((factor - mean_ewm) / (std_ewm + 1e-9)).clip(-3, 3)


    # # 过滤微弱信号（阈值可调）
    threshold = 0.1
    fct = _factor.fillna(0).where(lambda x: np.abs(x) >= threshold, 0)

    #再用tanh缩放，收敛到[-1,1]
    #fct = np.tanh(_factor / 3) * 3

    return fct.round(4)



def normalize_factor_quantile_discrete_vectorized(
    factor: pd.Series,
    window: int = 100,
    target_range=(-3, 3),
    step: float = 0.5
) -> pd.Series:
    """
    向量化实现：将因子按 rolling 分位数缩放到 [-3, 3] 并离散化（每 0.5 为一档）。
    """
    import pandas as pd
    import numpy as np

    n_bins = int((target_range[1] - target_range[0]) / step) + 1
    bin_edges = np.linspace(0, 1, n_bins)  # 分位边界
    bin_values = np.linspace(target_range[0], target_range[1], n_bins)  # 输出值

    # rolling rank：每个点在其窗口中的百分位
    def rolling_percentile(series):
        ranks = series.rank(pct=True).iloc[-1]  # 只保留最后一个时刻
        return ranks

    percentiles = factor.rolling(window=window, min_periods=window).apply(
        rolling_percentile, raw=False
    )

    # 离散化：将百分位 rank 映射到最近的 bin
    bin_indices = np.searchsorted(bin_edges, percentiles, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_values) - 1)

    scaled = pd.Series(bin_values[bin_indices], index=factor.index)
    return scaled



def discretize_factor(factor: pd.Series, step: float = 0.25) -> pd.Series:
    min_val, max_val = -3, 3

    # 使用apply函数对每个元素应用discretize_value函数
    def discretize_value(value):
        # 将因子值映射到离散化的范围内
        discretized_value = np.round((value - min_val) / step) * step + min_val
        # 确保结果在指定的范围内
        return min(max(discretized_value, min_val), max_val)

    return factor.apply(discretize_value)