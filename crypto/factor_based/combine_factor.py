from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from util.norm import normalize_factor, discretize_factor
from verify_risk_orthogonalization import risk_orthogonalization
import lightgbm as lgb

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
    X = df[factor_cols].values  # 风险正交化
    #print(X.columns)
    #X = df[factor_cols].values
    w = np.array(weights)
    combined = X @ w  # 矩阵乘法：每行做线性组合
    return pd.Series(combined, index=df.index, name="combined_factor")


def _compute_rolling_sharpe(series: pd.Series, window: int = 60) -> pd.Series:
    """
    计算滑动窗口内的年化 Sharpe ratio。
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(365)
    return sharpe


def combine_factors_lightgbm(df: pd.DataFrame, 
                             factor_cols: list, 
                             return_col: str = "log_return", 
                             lgbm_params: dict = None,
                             sharpe_window: int = 60*24,
                             **kwargs) -> pd.Series:
    """
    使用 LightGBM 学习一个非线性组合因子，以最大化未来的滑动年化 Sharpe。

    :param df: 包含因子与收益率的 DataFrame
    :param factor_cols: 要组合的因子列名
    :param return_col: 收益列（用于计算 Sharpe）
    :param lgbm_params: LightGBM 参数
    :param sharpe_window: 滑动窗口大小，用于计算 y = 年化 Sharpe
    :return: 组合因子（Series）
    """
    df = df.copy()

    # 计算未来收益的年化 Sharpe Ratio 作为标签 y
    df['future_sharpe'] = _compute_rolling_sharpe(df[return_col].shift(-1), window=sharpe_window).fillna(0)

    X = df[factor_cols]
    X = risk_orthogonalization(X)  # 风险正交化处理
    y = df['future_sharpe']

    # LightGBM 参数
    if lgbm_params is None:
        lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }

    print("训练目标：拟合未来 Sharpe 最大的因子组合。")
    model = lgb.LGBMRegressor(**lgbm_params)

    model.fit(X, y)

    # 提取特征重要性作为线性权重组合
    importance = model.feature_importances_
    weights = importance / importance.sum()

    combined_factor = X.dot(weights)
    combined_factor = pd.Series(combined_factor, index=df.index, name="combined_factor lightgmb")

    feature_importances = pd.Series(weights, index=X.columns, name="Feature Importance")
    print("\n模型学习到的特征重要性:\n",feature_importances)
    return combined_factor
