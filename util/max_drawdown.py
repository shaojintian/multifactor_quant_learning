import pandas as pd

def calculate_max_drawdown(net_values: pd.Series) -> float:
    """
    计算最大回撤

    参数:
        net_values: 策略净值序列

    返回:
        最大回撤值
    """

    peak = net_values.cummax()
    drawdown = (net_values - peak) / peak
    return drawdown.min()
