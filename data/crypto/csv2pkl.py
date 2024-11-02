import pandas as pd
import os

# 定义文件路径
csv_file_path = 'data/crypto/btcusdt_60m.csv'  # 替换为您的CSV文件路径
pkl_file_path = 'data/crypto/btcusdt_60m.pkl'  # 替换为您想要保存的PKL文件路径

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 将DataFrame保存为PKL文件
df.to_pickle(pkl_file_path)

print(f"CSV文件已成功转换为PKL文件: {pkl_file_path}")