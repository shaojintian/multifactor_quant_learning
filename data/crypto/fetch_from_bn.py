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
        current_start_time = klines_df['Close Time'].iloc[-1] + 1  # 加1毫秒以避免重复
        
    # 合并所有数据
    return pd.concat(all_klines, ignore_index=True)

# 示例调用
symbol = 'BTCUSDT'  # 交易对
interval = '1h'    # 时间间隔
start_time = date_to_utc_milliseconds("2018-01-01")  # 起始时间（毫秒）
end_time = date_to_utc_milliseconds("2024-01-01")    # 结束时间（毫秒）

# 获取所有K线数据
all_klines_df = fetch_all_klines(symbol, interval, start_time, end_time)

# 保存到CSV文件
all_klines_df.to_csv(f'data/crypto/btcusdt_{interval}.csv', index=False)
print(f"Data fetching complete. length: {len(all_klines_df)}")
