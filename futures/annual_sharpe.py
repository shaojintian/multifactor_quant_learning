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