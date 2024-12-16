import os
import pandas as pd

def load_data_from_directory(directory, time_period='3m'):
    data_frames = []
    
    # 遍历指定目录，加载每个币种的数据
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder, time_period)
        
        # 如果目录存在
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                
                # 假设文件是CSV格式的
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    
                    # 添加一个新列来标识这个币种
                    df['symbol'] = folder
                    data_frames.append(df)
    
    # 将所有的 DataFrame 合并到一个大的 DataFrame
    full_data = pd.concat(data_frames, ignore_index=True)
    
    return full_data

# 示例目录路径，假设数据存放在 '/mnt/data/data/futures/um/monthly/klines'
directory = '/Users/wanting/Downloads/multifactor_quant_learning/data/crypto/data/futures/um/monthly/klines'

# 加载数据
df = load_data_from_directory(directory)

# 显示合并后的数据
print(df.head())


