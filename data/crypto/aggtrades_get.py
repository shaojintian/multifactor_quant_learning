# import binance_bulk_downloader
from bulk_downloader import BinanceBulkDownloader

# download monthly aggTrades (timeperiod_per_file="monthly")
downloader = BinanceBulkDownloader(data_type="bookDepth",start_date="2024-01")
downloader.run_download()