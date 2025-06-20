import requests
import sys
import os
import pandas as pd

project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
sys.path.append(project_root)



from util.date import date_to_utc_milliseconds


def get_klines(symbol, interval, start_time=None, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000  # 最大返回条数
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    df.columns = [col.lower() for col in df.columns]
    return df

def fetch_all_klines(symbol, interval, start_time, end_time):
    all_klines = []
    current_start_time = start_time

    while current_start_time < end_time:
        print(f"Fetching data from {current_start_time} to {end_time}")
        klines_df = get_klines(symbol, interval, start_time=current_start_time, end_time=end_time)
        
        if klines_df.empty:
            break  # 如果没有数据，退出循环
        
        all_klines.append(klines_df)
        
        # 更新当前开始时间为最后一条数据的时间
        current_start_time = klines_df['close time'].iloc[-1] + 1  # 加1毫秒以避免重复
        
    # 合并所有数据
    return pd.concat(all_klines, ignore_index=True)

# 示例调用
symbol = 'ETHUSDT'  # 交易对
interval = '1h'    # 时间间隔
start_time = date_to_utc_milliseconds("2020-10-01")  # 起始时间（毫秒）
end_time = date_to_utc_milliseconds("2025-07-10")    # 结束时间（毫秒）

# 获取所有K线数据
all_klines_df = fetch_all_klines(symbol, interval, start_time, end_time)

# 保存到CSV文件
file_path = 'data/crypto/ethusdt_60m.csv'

if os.path.exists(file_path):
    existing_df = pd.read_csv(file_path, index_col=0)  # 读时把原索引当成index
    combined_df = pd.concat([existing_df, all_klines_df], ignore_index=False)  # 保留原索引，不重置
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # 按索引去重，保留第一个
    combined_df = combined_df.drop_duplicates(subset=['open time'], keep='first')  # 按 open time 去重
else:
    combined_df = all_klines_df

combined_df.index = combined_df["open time"]
combined_df.to_csv(file_path)
print(f"Data fetching complete. length: {len(all_klines_df)}")
