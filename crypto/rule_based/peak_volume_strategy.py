from matplotlib import pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from atomicx import AtomicFloat,AtomicInt
from dataclasses import dataclass
from typing import Optional
from .base_strategy import BaseStrategy

@dataclass
class Position:
    """Thread-safe position data structure"""
    amount:  AtomicFloat # 修改这里
    price: AtomicFloat   # 修改这里
    symbol: Optional[str]

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
        self._balance = AtomicInt(initial_balance)  # 修改这里
        self._position = Position(
            amount=AtomicFloat(0),  # 修改这里
            price=AtomicFloat(0),   # 修改这里
            symbol=None
        )
        self.balance_traces = pd.Series(
            [initial_balance],
            index=[pd.to_datetime('2024-11-01')]
        )
        self.lock = threading.Lock()  # Only used for balance_traces updates
    
    @property
    def balance(self):
        return self._balance.value

    @property
    def position(self):
        return self._position.amount.value

    def detect_rotation(self, volumes):
        """
        使用向量化操作检测轮动信号
        """
        if len(volumes) < self.volume_window + 1:
            return False
        
        moving_average_volume = volumes[:-1].tail(self.volume_window).mean()
        current_volume = volumes.iloc[-1]
        
        return current_volume > moving_average_volume * self.volume_multiplier[0]

    def buy(self, symbol: str, price: float, quote_asset_volume: float, current_time: int):
        """
        原子操作执行买入
        """
        # 计算可买入数量
        max_amount = min(
            self._balance.value * 0.99 / price,
            quote_asset_volume * 0.99 / price
        )
        
        # 原子更新持仓
        self._position.amount.set(max_amount)
        self._position.price.set(price)
        self._position.symbol = symbol
        
        # 原子更新余额
        self._balance.add(-max_amount * price)
        
        print(f"Buy: {symbol}, Price: {price:.2f}, Amount: {max_amount:.4f} "
              f"Time: {pd.to_datetime(current_time, unit='ms')}")
        
        self.record_balance(current_time)

    def sell(self, price: float, current_time: int):
        """
        原子操作执行卖出
        """
        # 获取当前持仓信息
        position_amount = self._position.amount.value
        symbol = self._position.symbol
        
        if position_amount > 0:
            # 计算卖出后的余额并原子更新
            self._balance.add(position_amount * price)
            
            # 清空持仓
            self._position.amount.set(0)
            self._position.price.set(0)
            self._position.symbol = None
            
            print(f"Sell: {symbol}, Price: {price:.2f}, "
                  f"Total Balance: {self._balance:.2f} "
                  f"Time: {pd.to_datetime(current_time, unit='ms')}")
            
            self.record_balance(current_time)

    def handle_symbol(self, symbol: str, df: pd.DataFrame):
        """
        优化后的单个symbol处理逻辑
        """
        # 预计算所需的数据以减少循环中的计算量
        closes = df['close'].values
        times = df['open_time'].values
        volumes = df['volume'].values
        quote_volumes = df['quote_volume'].values
        
        for idx in range(self.volume_window, len(df)):
            current_close = closes[idx]
            current_time = times[idx]
            current_quote_volume = quote_volumes[idx]
            
            # 使用切片获取历史成交量数据
            historical_volumes = pd.Series(volumes[:idx + 1])
            
            if self.detect_rotation(historical_volumes):
                current_position_symbol = self._position.symbol
                
                # 如果持有其他币种，先卖出
                if self.position > 0 and current_position_symbol != symbol:
                    self.sell(current_close, current_time)
                
                # 如果没有持仓，执行买入
                if self.position == 0:
                    self.buy(symbol, current_close, current_quote_volume, current_time)
        
        # 结束时平仓
        if self.position > 0 and self._position.symbol == symbol:
            self.sell(closes[-1], times[-1])

    def record_balance(self, current_time: int):
        """
        使用锁保护 DataFrame 操作
        """
        with self.lock:
            new_balance = pd.Series(
                [self._balance.value],
                index=[pd.to_datetime(current_time, unit='ms')]
            )
            self.balance_traces = pd.concat([self.balance_traces, new_balance])

    def compute_metrics(self, data):
        # Implement your metric computation logic
        pass

    def plot_balance(self, balance_history):
        # Implement your balance plotting logic
        plt.figure(figsize=(10, 6))
        plt.plot(self.balance_traces)
        plt.title('Shanzhai Rotation Strategy Balance')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.show()

    def run(self, data):
        # Implement your main strategy execution logic
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
        with self.lock:  # 使用锁确保线程安全
            new_balance = pd.Series([self.balance], index=[pd.to_datetime(current_time, unit='ms')])
            self.balance_traces = pd.concat([self.balance_traces, new_balance])

    def simulate_balance(self, signals, data):
        # Implement your balance simulation logic
        pass


    