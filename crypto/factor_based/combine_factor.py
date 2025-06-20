from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from util.norm import normalize_factor, discretize_factor
from verify_risk_orthogonalization import risk_orthogonalization
import lightgbm as lgb


import pandas as pd
import numpy as np

def _compute_rolling_sortino(
    return_series: pd.Series, 
    window: int = 90 * 24, 
    risk_free_rate_hourly: float = 0.0,
    periods_per_year: int = 365 * 24
) -> pd.Series:
    """
    计算滚动 Sortino 比率。

    Sortino 比率 = (年化收益率 - 年化无风险利率) / 年化下行波动率

    Args:
        return_series (pd.Series): 资产的收益率序列 (例如, 每小时的 return)。
        window (int): 计算滚动的窗口大小（单位：数据点个数）。
        risk_free_rate_hourly (float): 每小时的无风险利率或目标收益率（MAR）。
                                      通常对于加密货币设为0。
        periods_per_year (int): 用于年化计算的周期数。
                                如果是小时数据，则为 365 * 24。
                                如果是日数据，则为 252 或 365。

    Returns:
        pd.Series: 滚动 Sortino 比率的序列。
    """
    rolling_sortino = []

    for i in range(len(return_series)):
        # 如果数据点不足一个窗口，则填充 NaN
        if i < window:
            rolling_sortino.append(np.nan)
            continue

        # 1. 获取当前窗口的收益率
        window_ret = return_series.iloc[i - window:i]

        # 2. 计算年化收益率
        #   - 计算窗口内的平均每期收益率
        #   - 乘以每年的周期数进行年化
        annual_ret = window_ret.mean() * periods_per_year

        # 3. 计算年化无风险利率
        annual_risk_free_rate = risk_free_rate_hourly * periods_per_year

        # 4. 计算下行波动率 (Downside Deviation)
        #   - 找出所有低于无风险利率（或目标收益率）的收益
        downside_returns = window_ret[window_ret < risk_free_rate_hourly]
        
        #   - 如果没有下行收益，则下行波动为0
        if len(downside_returns) == 0:
            downside_deviation = 0
        else:
            #   - 计算这些下行收益的方差
            downside_variance = (downside_returns - risk_free_rate_hourly)**2
            #   - 取均值后开方，得到每期的下行标准差
            #   - 注意：这里用 len(window_ret) 而不是 len(downside_returns) 作为分母
            #     这是 Sortino 比率的标准定义之一，惩罚的是整体的下行风险。
            downside_std = np.sqrt(downside_variance.sum() / len(window_ret))
            #   - 将每期的下行标准差年化
            downside_deviation = downside_std * np.sqrt(periods_per_year)

        # 5. 计算 Sortino 比率
        #   - 处理分母为0的情况
        if downside_deviation > 1e-8: # 使用一个很小的数避免浮点精度问题
            excess_return = annual_ret - annual_risk_free_rate
            sortino = excess_return / downside_deviation
        else:
            # 如果下行波动为0，但有正的超额收益，可以认为是无限好，返回 inf
            # 如果超额收益也为0或负，则返回0
            if annual_ret > annual_risk_free_rate:
                sortino = np.inf
            else:
                sortino = 0.0
        
        rolling_sortino.append(sortino)
        
    return pd.Series(rolling_sortino, index=return_series.index)

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
    assert len(factor_cols) == len(weights), f"因子列数与权重长度不匹配{len(factor_cols)} != {len(weights)}"
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
def _compute_mdd(series: pd.Series, window: int = 30*24) -> pd.Series:
    """
    计算滑动窗口内的最大回撤（MDD）。
    返回的是每个点对应窗口内的最大回撤。
    """
    def calc_mdd(x):
        cummax = np.maximum.accumulate(x)
        drawdown = (x - cummax) / cummax
        return drawdown.min()  # 最小的回撤值（负值）

    return series.rolling(window).apply(calc_mdd, raw=True)




def _compute_rolling_calmar(return_series: pd.Series, window: int = 90*24) -> pd.Series:
    """
    计算滚动 Calmar 比率：年化收益率 / 最大回撤（负值）
    return_series 应为未来收益（例如 shift(-1)）
    """
    rolling_calmar = []

    for i in range(len(return_series)):
        if i < window:
            rolling_calmar.append(np.nan)
            continue

        window_ret = return_series.iloc[i - window:i]
        cum_ret = (1 + window_ret).cumprod()
        max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()  # 最大回撤（负数）
        annual_ret = window_ret.mean() * 365 * 24  # 假设日频，可以改为每小时 *24*365

        if max_dd < 0:
            calmar = annual_ret / abs(max_dd)
        else:
            calmar = 0

        rolling_calmar.append(calmar)

    return pd.Series(rolling_calmar, index=return_series.index)


def _compute_rolling_combined_score(return_series: pd.Series, window: int = 60*24) -> pd.Series:
    """
    计算滚动评分：Calmar（年化收益 / 最大回撤） + 0.5 * Sharpe
    return_series 应为未来收益（例如 shift(-1)）
    """
    scores = []

    for i in range(len(return_series)):
        if i < window:
            scores.append(np.nan)
            continue

        window_ret = return_series.iloc[i - window:i]
        cum_ret = (1 + window_ret).cumprod()
        max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()  # 最大回撤（负数）
        annual_ret = window_ret.mean() * 365 * 24  # 小时级别数据

        sharpe = 0
        if window_ret.std() > 0:
            sharpe = (window_ret.mean() / window_ret.std()) * np.sqrt(365 * 24)

        if max_dd < 0:
            calmar = annual_ret / abs(max_dd)
        else:
            calmar = 0

        # 组合目标函数
        score = calmar + 0.5 * sharpe
        scores.append(score)

    return pd.Series(scores, index=return_series.index)


def combine_factors_lightgbm(df: pd.DataFrame, 
                             factor_cols: list, 
                             return_col: str = "returns", 
                             lgbm_params: dict = None,
                             sharpe_window: int = 60*24,  # 默认60天
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
    #df['future_sharpe'] = _compute_rolling_sharpe(df[return_col].shift(-1), window=sharpe_window).fillna(0)
    #df['future_calmar'] = _compute_rolling_combined_score(df[return_col].shift(-1),window=sharpe_window).fillna(0)

    df['future_calmar'] =  _compute_rolling_calmar(df[return_col].shift(-1))  # LightGBM 参数
    X = df[factor_cols]
    X = risk_orthogonalization(X)  # 风险正交化处理
    y = df[return_col].shift(-1).rolling(window=3*24, min_periods=24).mean().fillna(0)

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

    X_train = X[:len(X)*4//5]  # 前 80% 数据作为训练集
    y_train = y[:len(y)*4//5]  # 前 80% 数据作为训练集
    model.fit(X_train, y_train)

    # raw_combined_factor = model.predict(X)

    # # 3. 转换为时间序列因子
    # combined_factor = pd.Series(raw_combined_factor, index=df.index, name="non_linear_factor")

    #提取特征重要性作为线性权重组合
    importance = model.feature_importances_
    weights = importance / importance.sum()

    combined_factor = X.dot(weights)
    combined_factor = pd.Series(combined_factor, index=df.index, name="combined_factor lightgmb")

    feature_importances = pd.Series(weights, index=X.columns, name="Feature Importance")
    print("\n模型学习到的特征重要性:\n",feature_importances.sort_values(ascending=False))

    combined_factor = combined_factor
    
    # combined_factor = np.tanh(combined_factor) * 1.5
    return combined_factor






