import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# 策略基类
class BaseStrategy(ABC):
    def __init__(self,balance=10000,data:pd.DataFrame=None):
        self.balance = balance
        self.data = data
    @abstractmethod
    def compute_metrics(self, prices, risk_free_rate=0.0):
        pass

    @abstractmethod
    def simulate_balance(self, prices, initial_balance=100000):
        pass

    @abstractmethod
    def plot_balance(self, balance):
        pass

    def compute_daily_returns(self, prices):
        return prices.pct_change().shift(-1).dropna()

    def calculate_sharpe_ratio(self, daily_returns:np.array, risk_free_rate=0.0):
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe_ratio

    def calculate_sortino_ratio(self, daily_returns:np.array, risk_free_rate=0.0):
        downside_returns = daily_returns[daily_returns < 0]
        expected_return = np.mean(daily_returns) - risk_free_rate
        downside_deviation = np.std(downside_returns)
        sortino_ratio = expected_return / downside_deviation * np.sqrt(252)
        return sortino_ratio

    def calculate_p_value(self, daily_returns):
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(daily_returns, 0)
        return p_value
