from backtest import BaseStrategy
from matplotlib import pyplot as plt

#✗ 山寨币轮动：当前1-3min成交额是之前序列时间窗口例如100个bar均值的50-100倍以上
#本质是跟随知情交易者或操纵市场

class ShanzhaiRotationStrategy(BaseStrategy):
    def __init__(self, volume_window=100, volume_multiplier=100):
        self.volume_window = volume_window
        self.volume_multiplier = volume_multiplier
    
    def compute_metrics(self, prices, volumes, risk_free_rate=0.0):
        """
        计算策略的指标，例如Sharpe比率、Sortino比率等。
        这个策略主要是基于成交量的异常波动。
        """
        daily_returns = self.compute_daily_returns(prices)
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns, risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(daily_returns, risk_free_rate)
        p_value = self.calculate_p_value(daily_returns)
        return sharpe_ratio, sortino_ratio, p_value

    def simulate_balance(self, prices, initial_balance=100000):
        daily_returns = self.compute_daily_returns(prices)
        balance = initial_balance * (1 + daily_returns).cumprod()
        return balance

    def plot_balance(self, balance):
        plt.figure(figsize=(10, 6))
        plt.plot(balance)
        plt.title('Shanzhai Rotation Strategy Balance')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.show()
    
    def detect_rotation(self, volumes):
        """
        检测是否出现山寨币轮动的交易信号
        当前成交量是之前100个bar均值的50-100倍以上
        """
        moving_average_volume = volumes[-self.volume_window:-1].mean()  # 过去100个bar的均值
        current_volume = volumes[-1]  # 当前的成交量
        if current_volume > moving_average_volume * self.volume_multiplier[0]:
            return True  # 检测到轮动信号
        return False

