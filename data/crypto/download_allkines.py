# import binance_bulk_downloader
from binance_bulk_downloader.downloader import BinanceBulkDownloader

# generate instance
downloader = BinanceBulkDownloader()

# download klines (frequency: "5m", asset="um")
downloader = BinanceBulkDownloader(data_frequency="3m")
downloader.run_download()