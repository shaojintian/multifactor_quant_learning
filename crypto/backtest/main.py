import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 计算日收益率
def compute_daily_returns(prices):
    return prices.pct_change().dropna()

# 计算Sharpe比率
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    excess_returns = daily_returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 年化
    return sharpe_ratio

# 计算Sortino比率
def calculate_sortino_ratio(daily_returns, risk_free_rate=0.0):
    downside_returns = daily_returns[daily_returns < 0]
    expected_return = np.mean(daily_returns) - risk_free_rate
    downside_deviation = np.std(downside_returns)
    sortino_ratio = expected_return / downside_deviation * np.sqrt(252)  # 年化
    return sortino_ratio

# 计算p-value (使用t检验)
def calculate_p_value(daily_returns):
    t_stat, p_value = stats.ttest_1samp(daily_returns, 0)  # 对于零假设：均值为0
    return p_value

# 模拟账户余额
def simulate_account_balance(daily_returns, initial_balance=100000):
    balance = initial_balance * (1 + daily_returns).cumprod()  # 账户余额模拟
    return balance

# 绘制账户余额图
def plot_account_balance(balance):
    plt.figure(figsize=(10, 6))
    plt.plot(balance)
    plt.title('Account Balance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.grid(True)
    plt.show()

# 主回测函数
def backtest(prices, initial_balance=100000, risk_free_rate=0.0):
    # 计算日收益率
    daily_returns = compute_daily_returns(prices)

    # 计算Shapre比率
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)

    # 计算Sortino比率
    sortino_ratio = calculate_sortino_ratio(daily_returns, risk_free_rate)

    # 计算p-value
    p_value = calculate_p_value(daily_returns)

    # 模拟账户余额
    account_balance = simulate_account_balance(daily_returns, initial_balance)

    # 打印结果
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # 绘制账户余额曲线
    plot_account_balance(account_balance)

    return sharpe_ratio, sortino_ratio, p_value, account_balance

# 示例数据：使用某只股票的历史收盘价数据
# 请替换为实际的数据，例如使用yfinance下载股票数据
# 示例中使用随机数据作为示范
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2021-12-31', freq='B')  # 工作日数据
prices = pd.Series(np.random.rand(len(dates)) * 100 + 1000, index=dates)

# 执行回测
sharpe_ratio, sortino_ratio, p_value, account_balance = backtest(prices)
