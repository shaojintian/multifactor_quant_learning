from matplotlib import pyplot as plt
from .base_strategy import BaseStrategy
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
#✗ 山寨币轮动：当前1-3min成交额是之前序列时间窗口例如100个bar均值的50-100倍以上
#本质是跟随知情交易者或操纵市场

class ShanzhaiRotationStrategy(BaseStrategy):
    def __init__(self, volume_window=100, volume_multiplier=(5, 100), initial_balance=10000):
        """
        初始化轮动策略
        :param volume_window: 成交量均值计算窗口
        :param volume_multiplier: 成交量倍数阈值 (min_multiplier, max_multiplier)
        :param initial_balance: 初始账户余额
        """
        self.volume_window = volume_window
        self.volume_multiplier = volume_multiplier
        self.balance = initial_balance
        self.position = 0  # 当前仓位
        self.asset_price = 0  # 买入时的价格
        self.symbol = None  # 当前持仓的币种
        self.balance_traces =pd.Series([self.balance],index=datetime.datetime(2024,11,0,0,0)) # 账户余额记录
        
    
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

    def plot_balance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.balance_traces)
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

    def detect_rotation(self, volumes):
        """
        检测是否出现山寨币轮动的交易信号
        当前成交量是过去 volume_window 个 bar 均值的 50-100 倍以上
        """
        if len(volumes) < self.volume_window + 1:  # 数据不足，返回 False
            return False
        
        moving_average_volume = volumes[-self.volume_window-1:-1].mean()
        current_volume = volumes.iloc[-1]
        
        if current_volume > moving_average_volume * self.volume_multiplier[0]:
            return True  # 检测到轮动信号
        return False

    def buy(self, symbol, price):
        """
        模拟买入操作
        """
        amount_to_buy = self.balance / price
        self.position = amount_to_buy
        self.asset_price = price
        self.symbol = symbol
        self.balance -=  amount_to_buy * self.asset_price # 全仓买入
        print(f"Buy: {symbol}, Price: {price:.2f}, Amount: {amount_to_buy:.4f}")

    def sell(self, price):
        """
        模拟卖出操作
        """
        self.balance = self.position * price
        print(f"Sell: {self.symbol}, Price: {price:.2f}, Total: {self.balance:.2f}")
        self.position = 0
        self.asset_price = 0
        self.symbol = None

    def run(self, data):
        """
        遍历所有 symbol 的数据，执行轮动检测并交易
        """
        #print("Running Shanzhai Rotation Strategy ...")
        for symbol, df in data.items():
            for idx in range(self.volume_window, len(df)):
                current_close= df['close'].iloc[idx]  # 获取当前时间点的数据
                current_time = df['open_time'].iloc[idx]
                volumes = df['volume'][:idx+1]  # 获取到当前时间点的所有成交量数据
                # 检测轮动信号
                if self.detect_rotation(volumes):
                    # 如果已有持仓且当前 symbol 不同，先卖出
                    if self.position > 0 and self.symbol != symbol:
                        self.sell(current_close)
                        self.record_balance(current_time)
                        
                    # 如果没有持仓，执行买入
                    if self.position == 0:
                        self.buy(symbol, current_close)
                        self.record_balance(current_time)
            
            # 如果最后仍有持仓，假设在最后时刻平仓
            if self.position > 0:
                self.sell(current_close)
                self.record_balance(current_time)
            
            #
            self.plot_balance()
            print("Finished Shanzhai Rotation Strategy")
        

    def record_balance(self,current_time):
        # 记录账户余额
        new_balance = pd.Series([self.balance], index=current_time)

        # Assuming self.balance_traces is the existing Series
        self.balance_traces = pd.concat([self.balance_traces, new_balance])

