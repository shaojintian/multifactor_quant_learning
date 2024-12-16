import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rule_based.peak_volume_strategy import ShanzhaiRotationStrategy
from data.crypto.data.futures.um.monthly.klines import *


class StrategyFactory:
    def __init__(self):
        self.strategies = {}

    def register_strategy(self, name, strategy_class):
        """注册一个策略"""
        self.strategies[name] = strategy_class

    def get_strategy(self, name):
        """根据策略名称获取策略"""
        strategy_class = self.strategies.get(name)
        if not strategy_class:
            raise ValueError(f"Strategy {name} not registered!")
        return strategy_class()

    def run_strategy(self, name, data:pd.DataFrame, initial_balance=100000, risk_free_rate=0.02):
        """运行一个策略"""
        strategy = self.get_strategy(name)
        return strategy.run(data)


# Example usage
if __name__ == "__main__":
    # 创建策略工厂
    factory = StrategyFactory()

    # 注册策略
    print("Registering strategies...")
    factory.register_strategy("ShanzhaiRotationStrategy", ShanzhaiRotationStrategy)

    # 创建示例数据（可以替换为真实数据）
   

    # 执行策略
    factory.run_strategy("ShanzhaiRotationStrategy", data)

