import akshare as ak
import pandas as pd
from tqsdk import TqApi, TqAuth


symbols = []
_periods = 30
def fetch_data(_symbol, _periods):
    # Fetch data from the commodities_data.csv file
    # and return it as a pandas DataFrame
    futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol=_symbol, period=_periods)
    futures_zh_minute_sina_df.to_csv(f'akshare_{_symbol}_{_periods}.csv', index=False)
    return futures_zh_minute_sina_df


if __name__ == "__main__":
    fetch_data("IC888", _periods)

