import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data/crypto/BTC_USDT_day.csv')

# 将列标题转换为小写
df.columns = df.columns.str.lower()

df.to_csv('data/crypto/BTC_USDT_day.csv')