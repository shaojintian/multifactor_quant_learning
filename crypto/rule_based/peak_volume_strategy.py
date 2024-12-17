from matplotlib import pyplot as plt
from .base_strategy import BaseStrategy
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
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
        self.balance_traces =pd.Series([self.balance],index=[pd.to_datetime('2024-11-01')]) # 账户余额记录
        
        # 线程锁，保护共享变量
        self.lock = threading.Lock()
    
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
        当前成交量是过去 volume_window 个 bar 均值的 50-100 倍以上
        """
        if len(volumes) < self.volume_window + 1:  # 数据不足，返回 False
            return False
        
        moving_average_volume = volumes[:-1].tail(self.volume_window).mean()
        current_volume = volumes.iloc[-1]
        
        if current_volume > moving_average_volume * self.volume_multiplier[0]:
            return True  # 检测到轮动信号
        return False

    def buy(self, symbol, price,quote_asset_volume,current_time):
        """
        模拟买入操作
        """
        with self.lock:  # 使用锁确保线程安全
            amount_to_buy = min(self.balance*0.99/price, quote_asset_volume*0.99/price) 
            self.position = amount_to_buy
            self.asset_price = price
            self.symbol = symbol
            self.balance -= amount_to_buy * price  # 全仓买入
        print(f"Buy: {symbol}, Price: {price:.2f}, Amount: {amount_to_buy:.4f} Time: {pd.to_datetime(current_time, unit='ms')}")

    def sell(self, price,current_time):
        """
        模拟卖出操作
        """
        with self.lock:  # 使用锁确保线程安全
            self.balance = self.position * price
            self.position = 0
            self.asset_price = 0
            self.symbol = None
        print(f"Sell: {self.symbol}, Price: {price:.2f}, Total Balance: {self.balance:.2f} Time: {pd.to_datetime(current_time, unit='ms')}")

    def handle_symbol(self, symbol, df):
        """
        处理单个 symbol 的买卖逻辑
        """
        for idx in range(self.volume_window, len(df)):
            with self.lock:  # 确保在修改共享资源时使用锁
                current_close = df['close'].iloc[idx]  # 获取当前时间点的数据
                current_time = df['open_time'].iloc[idx]
                current_quote_asset_volume = df['quote_volume'].iloc[idx]
                volumes = df['volume'][:idx + 1]  # 获取到当前时间点的所有成交量数据

                # 检测轮动信号
                if self.detect_rotation(volumes):
                    # 如果已有持仓且当前 symbol 不同，先卖出
                    if self.position > 0 and self.symbol != symbol:
                        self.sell(current_close, current_time)
                    
                    # 如果没有持仓，执行买入
                    if self.position == 0:
                        self.buy(symbol, current_close, current_quote_asset_volume, current_time)
        
        # 如果最后仍有持仓，假设在最后时刻平仓
        if self.position > 0:
            self.sell(current_close, current_time)
            self.record_balance(current_time)

    def run(self, data):
        """
        遍历所有 symbol 的数据，执行轮动检测并交易
        """
        print("Running Shanzhai Rotation Strategy ...")
        
        with ThreadPoolExecutor() as executor:  # 使用线程池执行并发任务
            futures = []
            for symbol, df in data.items():
                # 提交每个 symbol 的处理任务
                futures.append(executor.submit(self.handle_symbol, symbol, df))
            
            # 等待所有任务完成
            for future in futures:
                future.result()

        self.plot_balance()
        print("Finished Shanzhai Rotation Strategy")

    def record_balance(self,current_time):
        new_balance = pd.Series([self.balance], index=[pd.to_datetime(current_time, unit='ms')])
        with self.lock:  # 使用锁确保线程安全
            self.balance_traces = pd.concat([self.balance_traces, new_balance])

