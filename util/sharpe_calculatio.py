# 4 .计算annual夏普比率
import numpy as np

# 输入为每日净值序列
def cal_sharp(net_values: np.array) -> float:
    '''计算夏普比率，risk free rate为无风险年化收益率，trading_days是1年的交易日'''
    risk_free_rate = 0.03  # 无风险收益率
    trading_days = 252  # 一年的交易日

    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]  # 计算相对收益率
    mean_return = np.mean(returns)  # 平均收益率

    # 计算超额收益率
    excess_return = mean_return - risk_free_rate  # 超额收益率

    # 计算收益率的标准差
    std_dev = np.std(returns,ddof = 1)  # 收益率的标准差

    # 计算夏普比率
    sharpe_ratio = (excess_return * trading_days) / std_dev  # 年化夏普比率

    return sharpe_ratio

#all day trading
def cal_annual_sharpe_15mins(net_values: np.array, risk_free_rate: float = 0.05) -> float:
    # 计算15分钟收益率
    returns = np.diff(net_values) / net_values[:-1]

    # 计算日收益率
    daily_return = (1 + returns) ** 96 - 1

    # 计算年化收益率
    annual_return = (1 + np.mean(daily_return)) ** 252 - 1

    # 计算年化标准差
    annual_std_dev = np.std(returns) * np.sqrt(252 * 96)

    # 计算超额收益
    excess_return = annual_return - risk_free_rate

    # 计算年化夏普比率
    if annual_std_dev == 0:
        return np.nan  # 避免除以零
    sharpe_ratio = excess_return / annual_std_dev

    return sharpe_ratio



def cal_sharp_random(net_values: np.array, period_minutes: int = 15, trading_hours: int = 4) -> float:
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
    annual_factor = periods_per_day * trading_days  # 年化系数
    
    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]
    
    # 确保returns不为空
    if len(returns) == 0:
        return np.nan
    
    # 计算年化收益率和年化标准差
    mean_return = np.mean(returns) * annual_factor
    std_dev = np.std(returns, ddof=1) * np.sqrt(annual_factor)
    
    # 确保标准差不为零
    if std_dev == 0:
        return np.nan
        
    # 计算超额收益率
    excess_return = mean_return - risk_free_rate
    
    # 计算夏普比率
    sharpe_ratio = excess_return / std_dev
    
    return sharpe_ratio