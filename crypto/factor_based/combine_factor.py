from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from util.norm import normalize_factor, discretize_factor

def combine_factors_by_regression(df: pd.DataFrame, factor_cols: list, ret: pd.Series, window: int = 100) -> pd.Series:
    """
    用滚动线性回归方式组合多个因子
    :param df: 包含所有因子和收益的 DataFrame
    :param factor_cols: 要组合的因子列名列表
    :param ret_col: 收益列名
    :param window: 回归窗口长度
    :return: 回归组合因子（Series）
    """
    combined = pd.Series(index=df.index, dtype=float)

    for i in range(window, len(df)):
        X = df[factor_cols].iloc[i-window:i]
        y = ret.iloc[i - window:i].values

        # 回归（自动处理多因子权重）
        model = LinearRegression().fit(X, y)
        weights = model.coef_

        # 当前时刻组合因子值
        current_factors = df[factor_cols].iloc[i]
        combined.iloc[i] = np.dot(current_factors.values, weights)

    combined = normalize_factor(combined)
    return combined

def combine_factors_by_discretization(df: pd.DataFrame, factor_cols: list, ret: pd.Series, window: int = 100) -> pd.Series:
    """
    用滚动线性回归方式组合多个因子
    :param df: 包含所有因子和收益的 DataFrame
    :param factor_cols: 要组合的因子列名列表
    :param ret_col: 收益列名
    :param window: 回归窗口长度
    :return: 回归组合因子（Series）
    """
    combined = pd.Series(index=df.index, dtype=float)

    for i in range(window, len(df)):
        X = df[factor_cols].iloc[i-window:i]
        y = ret.iloc[i - window:i].values

        # 回归（自动处理多因子权重）
        model = LinearRegression().fit(X, y)
        weights = model.coef_

        # 当前时刻组合因子值
        current_factors = df[factor_cols].iloc[i]
        combined.iloc[i] = np.dot(current_factors.values, weights)

    combined = normalize_factor(combined)
    combined = discretize_factor(combined)
    return combined

def combine_factors_linear(df: pd.DataFrame, factor_cols: list, weights: list) -> pd.Series:
    """
    简单线性组合多个因子：w1*x1 + w2*x2 + ...
    :param df: 包含因子的 DataFrame
    :param factor_cols: 要组合的因子列名
    :param weights: 与因子一一对应的权重列表
    :return: 组合因子（Series）
    """
    X = df[factor_cols].values
    w = np.array(weights)
    combined = X @ w  # 矩阵乘法：每行做线性组合
    return pd.Series(combined, index=df.index, name="combined_factor")
