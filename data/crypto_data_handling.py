import os
import pandas as pd
import sys,os
project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
sys.path.append(project_root)

def convert_column_names_to_lowercase(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 将列名转换为小写
            df.columns = df.columns.str.lower()
            
            # 将修改后的 DataFrame 保存回 CSV 文件
            df.to_csv(file_path, index=False)
            print(f"Processed file: {filename}")

# 指定 crypto 文件夹的路径
folder_path = 'data/crypto'  # 请根据实际路径修改
convert_column_names_to_lowercase(folder_path)