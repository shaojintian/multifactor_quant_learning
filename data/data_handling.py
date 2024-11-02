import os

def rename_files_to_lowercase(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 构建完整的文件路径
        old_file = os.path.join(directory, filename)
        
        # 检查是否是文件
        if os.path.isfile(old_file):
            # 生成新的文件名（小写）
            new_file = os.path.join(directory, filename.lower())
            
            # 重命名文件
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')

# 指定 crypto 文件夹的路径
crypto_directory = 'path/to/crypto'  # 替换为实际路径

# 调用函数
rename_files_to_lowercase(crypto_directory)