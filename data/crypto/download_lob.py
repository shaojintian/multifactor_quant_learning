# import binance_bulk_downloader
from bulk_downloader import BinanceBulkDownloader

# generate instance
downloader = BinanceBulkDownloader()


# download klines (frequency: "5m", asset="um")
downloader = BinanceBulkDownloader(data_type="bookDepth",
                                  asset="um", timeperiod_per_file="monthly",start_date="2020-01")
downloader.run_download()

'''
[
  [
    1499040000000,      // k线开盘时间
    "0.01634790",       // 开盘价
    "0.80000000",       // 最高价
    "0.01575800",       // 最低价
    "0.01577100",       // 收盘价(当前K线未结束的即为最新价)
    "148976.11427815",  // 成交量
    1499644799999,      // k线收盘时间
    "2434.19055334",    // 成交额
    308,                // 成交笔数
    "1756.87402397",    // 主动买入成交量
    "28.46694368",      // 主动买入成交额
    "17928899.62484339" // 请忽略该参数
  ]
]
GET /api/v3/klines

每根K线代表一个交易对。
每根K线的开盘时间可视为唯一ID

权重(IP): 2

参数:

名称	类型	是否必需	描述
symbol	STRING	YES	
interval	ENUM	YES	详见枚举定义：K线间隔
startTime	LONG	NO	
endTime	LONG	NO	
timeZone	STRING	NO	默认: 0 (UTC)
'''
