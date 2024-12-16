import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/Users/wanting/Downloads/multifactor_quant_learning")
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
        print(f"Running {name}...\n")
        strategy = self.get_strategy(name)
        return strategy.run(data)
    

    def load_data_from_csv(self,directory,start_date:str="2024-11"):
        data = {}
        
        # 遍历目录下的所有文件夹和 CSV 文件
        for symbol_folder in os.listdir(directory):
            symbol_path = os.path.join(directory, symbol_folder, '3m')  # 根据路径调整
            if os.path.isdir(symbol_path):  # 确保是目录
                for file_name in os.listdir(symbol_path):
                    if file_name.endswith(start_date+'.csv'):
                        file_path = os.path.join(symbol_path, file_name)
                        
                        # 提取symbol，假设文件名格式为 symbol-时间周期-其他信息.csv
                        symbol = file_name.split('-')[0]  # 通过'-'分隔并提取symbol部分
                        
                        df = pd.read_csv(file_path)
                        df.columns=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'Number of Trades',
                            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
                        ]
                        data[symbol] = df
                        #print(f"Loaded data for {symbol} from {file_path}")
        print("data loaded")
        return data



# Example usage
if __name__ == "__main__":
    #
    _start_date="2024-11"

    # 创建策略工厂
    factory = StrategyFactory()

    # 注册策略
    print("Registering strategies...")
    factory.register_strategy("ShanzhaiRotationStrategy", ShanzhaiRotationStrategy)

    # 创建示例数据（可以替换为真实数据）
    data :dict = factory.load_data_from_csv("/Users/wanting/Downloads/multifactor_quant_learning/data/crypto/data/futures/um/monthly/klines",start_date=_start_date)

    # 执行策略
    factory.run_strategy("ShanzhaiRotationStrategy", data)

