# 4 .计算annual夏普比率
import numpy as np

# 输入为每日净值序列
import numpy as np

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
