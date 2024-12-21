# import binance_bulk_downloader
from binance_bulk_downloader.downloader import BinanceBulkDownloader

# generate instance
downloader = BinanceBulkDownloader(data_type="aggTrades",start_date="2024-11")

# download aggTrades (asset="um")
downloader.run_download()

# download monthly aggTrades (timeperiod_per_file="monthly")
downloader = BinanceBulkDownloader(data_type="aggTrades", timeperiod_per_file="monthly")
downloader.run_download()

# download aggTrades (asset="cm")
downloader = BinanceBulkDownloader(data_type="aggTrades", asset="cm")
downloader.run_download()

# download monthly aggTrades (asset="cm", timeperiod_per_file="monthly")
downloader = BinanceBulkDownloader(
    data_type="aggTrades", asset="cm", timeperiod_per_file="monthly"
)
downloader.run_download()


'''
[
  {
    a: 416690, // 归集成交ID
    p: "9642.4", // 成交价
    q: "3", // 成交量
    f: 595259, // 被归集的首个成交ID
    l: 595259, // 被归集的末个成交ID
    T: 1591250548649, // 成交时间
    m: true, // 是否为主动卖出单
  }
]
'''