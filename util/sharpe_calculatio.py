# 4 .计算annual夏普比率
import numpy as np

# 输入为每日净值序列
import pandas as pd

def cal_sharp(net_values: np.array) -> float:
    '''计算夏普比率，risk_free_rate为无风险年化收益率，trading_days是1年的交易日'''
    risk_free_rate = 0.03  # 无风险收益率
    trading_days = 252  # 一年的交易日

    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]  # 计算相对收益率
    mean_return = np.mean(returns)  # 平均收益率

    # 计算超额收益率
    excess_return = mean_return - (risk_free_rate / trading_days)  # 每日超额收益率

    # 计算收益率的标准差
    std_dev = np.std(returns, ddof=1)  # 收益率的标准差

    # 计算年化夏普比率
    sharpe_ratio = (excess_return * np.sqrt(trading_days)) / std_dev  # 年化夏普比率

    return sharpe_ratio




def cal_sharp_random(net_values: np.array, period_minutes: int = 15, trading_hours: int = 6) -> float:
    '''计算年化夏普比率
    Args:
        net_values: np.array, 净值序列
        period_minutes: int, 数据周期（分钟）
        trading_hours: int, 每天交易小时数
    Returns:
        float: 年化夏普比率
    '''
    risk_free_rate = 0.03  # 年化无风险收益率
    
    # 计算年化系数
    periods_per_day = (trading_hours * 60) // period_minutes  # 每天的周期数
    trading_days = 252  # 一年的交易日数
    annual_factor = np.sqrt(periods_per_day * trading_days)  # 年化系数，使用根号下的总周期数
    
    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]
    
    # 确保returns不为空
    if len(returns) == 0:
        return np.nan
    
    # 计算平均收益率和标准差
    mean_return = np.mean(returns)
    std_dev = np.std(returns, ddof=1)
    
    # 确保标准差不为零
    if std_dev == 0:
        return np.nan
        
    # 计算年化收益率和年化标准差
    annualized_mean_return = mean_return * periods_per_day
    annualized_std_dev = std_dev * annual_factor
    
    # 计算超额收益率
    excess_return = annualized_mean_return - risk_free_rate
    
    # 计算夏普比率
    sharpe_ratio = excess_return / annualized_std_dev
    
    return sharpe_ratio


def calculate_sharpe_ratio_corrected(
    net_values: np.ndarray, 
    period_minutes: int, 
    trading_hours: int = 24
) -> float:
    """
    计算年化夏普比率 (修正并采用标准方法)
    
    Args:
        net_values (np.ndarray): 净值序列。
        period_minutes (int): 数据周期（分钟）。
        trading_hours (int): 每天交易小时数 (例如股票为4，加密货币为24)。
        
    Returns:
        float: 年化夏普比率。
    """
    # 1. 计算每个周期的收益率
    # 使用 pd.Series 可以更方便地处理百分比变化
    returns = pd.Series(net_values).pct_change().dropna()
    
    if len(returns) < 2:  # 需要至少2个收益率数据点来计算标准差
        return np.nan

    # 2. 计算年化系数
    # 确保 trading_hours * 60 不为0，避免除零错误
    if period_minutes == 0:
        return np.nan
    periods_per_day = (trading_hours * 60) / period_minutes
    trading_days = 365  # 传统金融市场交易日，对于加密货币可使用365
    annual_factor = np.sqrt(periods_per_day * trading_days)
    
    # 3. 计算周期性的平均收益率和标准差
    mean_return_periodic = returns.mean()
    std_dev_periodic = returns.std(ddof=1) # 使用样本标准差 (ddof=1)
    
    if std_dev_periodic == 0:
        # 如果波动率为0，但收益为正，理论上夏普为无穷大，这里返回一个大数值或np.inf
        # 如果收益也为0或负，返回0或nan
        return np.inf if mean_return_periodic > 0 else 0.0

    # 4. 计算周期性的无风险利率
    annual_risk_free_rate = 0.03  # 年化无风险收益率
    # 将年化无风险利率转换为当前周期的无风险利率
    risk_free_rate_periodic = annual_risk_free_rate / (periods_per_day * trading_days)

    # 5. 计算周期性的超额收益
    excess_return_periodic = mean_return_periodic - risk_free_rate_periodic
    
    # 6. 计算周期性的夏普比率
    sharpe_ratio_periodic = excess_return_periodic / std_dev_periodic
    
    # 7. 年化夏普比率
    annualized_sharpe_ratio = sharpe_ratio_periodic * annual_factor
    
    return annualized_sharpe_ratio

def calculate_calmar_ratio(net_value: pd.Series, periods_per_year: int = 365*24) -> float:
    """
    计算 Calmar Ratio（年化收益 / 最大回撤）

    参数:
        net_value (pd.Series): 净值序列，index 应为时间顺序。
        periods_per_year (int): 年度交易周期数，默认 365（适用于加密货币小时线）

    返回:
        float: Calmar 比率
    """
    # 计算收益率
    returns = net_value.pct_change().dropna()

    # 年化收益率（单利或复利都可以，此处用复利）
    total_return = net_value.iloc[-1] / net_value.iloc[0] - 1
    years = len(net_value) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1

    # 最大回撤
    peak = net_value.cummax()
    drawdown = (net_value - peak) / peak
    max_drawdown = drawdown.min()  # 是负值

    # Calmar Ratio
    if max_drawdown == 0:
        return np.nan  # 避免除以0
    return annual_return / abs(max_drawdown)


def calculate_annualized_return(net_value: pd.Series, periods_per_year: int = 365*24) -> float:
    """
    根据策略的净值序列计算年化收益率。
    Generated code
    该计算基于几何平均收益，公式为：
    (期末净值 / 期初净值) ^ (1 / 年数) - 1

    Args:
        net_value (pd.Series): 包含策略净值的时间序列。索引可以是时间，也可以是简单的数字序列。
        periods_per_year (int): 每年的周期数。
                                - 对于日度数据，通常使用 252 (交易日) 或 365 (日历日)。
                                - 对于小时数据，通常使用 252 * 4 (A股交易小时) 或 365 * 24。
                                - 对于分钟数据，则为 252 * 4 * 60。
                                默认为 252 (适用于日度数据)。

    Returns:
        float: 策略的年化收益率。如果数据点少于2个，则返回 0.0。
    """
    # 确保有足够的数据点进行计算
    if len(net_value) < 2:
        return 0.0

    # 计算总的年数
    # 假设数据点是均匀分布的
    num_years = len(net_value) / periods_per_year

    # 如果总时间跨度为0，无法计算，返回0
    if num_years == 0:
        return 0.0

    # 计算总收益率 (期末净值 / 期初净值)
    total_return_factor = net_value.iloc[-1] / net_value.iloc[0]

    # 使用几何平均值计算年化收益率
    # 公式: (总收益率) ^ (1 / 年数) - 1
    annualized_return = total_return_factor ** (1 / num_years) - 1

    return annualized_return
