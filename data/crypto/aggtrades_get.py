# import binance_bulk_downloader
from bulk_downloader import BinanceBulkDownloader

# download monthly aggTrades (timeperiod_per_file="monthly")
downloader = BinanceBulkDownloader(data_type="bookTicker",timeperiod_per_file="monthly",lable="BTCUSDT",start_date="2020-01",asset="um")
downloader.run_download()