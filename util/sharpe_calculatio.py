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