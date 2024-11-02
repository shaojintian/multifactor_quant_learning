import akshare as ak
import pandas as pd
from io import StringIO

# 下载中证500股指期货1小时数据
csi500_futures_data = ak.futures_main_sina(symbol="IC0", start_date="20160101", end_date="20240101")

# 查看数据
print(csi500_futures_data.head())

# 保存为 CSV 文件
csi500_futures_data.to_csv("commodities_data/csi500_futures_1d_data.csv", index=0)

# Check if data_json is valid before reading it
if csi500_futures_data.empty:  # Ensure csi500_futures_data is not empty
    print("Error: Received empty or invalid JSON data.")

# futures_display_main_sina_df = ak.futures_display_main_sina()
# print(futures_display_main_sina_df)