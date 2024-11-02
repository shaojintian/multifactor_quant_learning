import requests
import sys,os
project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
sys.path.append(project_root)
import pandas as pd
from util.date import date_to_utc_milliseconds


def get_klines(symbol, interval, start_time=None, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time
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

# 示例调用
symbol = 'BTCUSDT'  # 交易对
interval = '30m'     # 时间间隔
startTime =  date_to_utc_milliseconds("2018-01-01") # 起始时间（毫秒）
end_time = date_to_utc_milliseconds("2024-01-01")  # 结束时间（毫秒）
klines_df = get_klines(symbol, interval)

klines_df.to_csv(f'data/crypto/btcusdt_{interval}.csv', index=0)
