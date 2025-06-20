import os
import pandas as pd
import sys,os
project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
sys.path.append(project_root)

def convert_column_names_to_lowercase(folder_path):
    # 遍历文件夹及其所有子目录
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)
                    df.columns = df.columns.str.lower()
                    df.to_csv(file_path, index=False)
                    print(f"Processed file: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

# 替换为你的目标路径
folder_path = '/Users/wanting/Downloads/multifactor_quant_learning/data/crypto'


def concat_files_in_folder(folder_path):
    #TODO
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        print("No CSV files found in the specified folder.")
        return None
    
    combined_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    return combined_df


if __name__ == "__main__":
    # 先转换列名为小写
    convert_column_names_to_lowercase(folder_path)
    #concat_files_in_folder(folder_path)