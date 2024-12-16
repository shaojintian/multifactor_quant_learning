import pandas as pd
import numpy as np
from rule_based import ShanzhaiRotationStrategy

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
        prices = data['close']
        sharpe_ratio, sortino_ratio, p_value = strategy.compute_metrics(prices, risk_free_rate)
        balance = strategy.simulate_balance(prices, initial_balance)
        strategy.plot_balance(balance)
        print(f"Strategy: {name}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Sortino Ratio: {sortino_ratio:.4f}")
        print(f"P-Value: {p_value:.4f}")
        return sharpe_ratio, sortino_ratio, p_value, balance


# Example usage
if __name__ == "__main__":
    # 创建策略工厂
    factory = StrategyFactory()

    # 注册策略
    factory.register_strategy("ShanzhaiRotationStrategy", ShanzhaiRotationStrategy)

    # 创建示例数据（可以替换为真实数据）
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='B')
    data = pd.Series(np.random.rand(len(dates)) * 100 + 1000, index=dates)

    # 执行策略
    factory.run_strategy("ShanzhaiRotationStrategy", data)
