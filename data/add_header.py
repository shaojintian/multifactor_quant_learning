import pandas as pd

# 加载现有的CSV文件
df = pd.read_csv('your_file.csv')

# 假设您已经有了这些新列的数据，可以添加它们
# 如果这些数据是固定的值，可以这样添加：
df['quote asset volume'] = 'your_value'
df['number of trades'] = 'your_value'
df['taker buy base asset volume'] = 'your_value'
df['taker buy quote asset volume'] = 'your_value'

# 如果这些数据是根据某些计算得出的，您需要先进行计算，然后再添加列
# 例如，这里用随机数来模拟数据
df['quote asset volume'] = pd.Series(np.random.rand(len(df)))
df['number of trades'] = pd.Series(np.random.randint(1, 100, len(df)))
df['taker buy base asset volume'] = pd.Series(np.random.rand(len(df)))
df['taker buy quote asset volume'] = pd.Series(np.random.rand(len(df)))

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('updated_file.csv', index=False)
