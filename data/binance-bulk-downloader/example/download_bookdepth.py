# import binance_bulk_downloader
from binance_bulk_downloader.downloader import BinanceBulkDownloader

# generate instance
downloader = BinanceBulkDownloader(data_type="bookDepth")

# download bookDepth
downloader.run_download()

# download bookDepth (asset="cm")
downloader = BinanceBulkDownloader(data_type="bookDepth", asset="cm")
downloader.run_download()


'''
{
  "lastUpdateId": 16769853,
  "symbol": "BTCUSD_PERP", // 交易对
  "pair": "BTCUSD",		 // 标的交易对
  "E": 1591250106370,   // 消息时间
  "T": 1591250106368,   // 撮合时间
  "bids": [				 // 买单
    [
      "9638.0",     	// 价格
      "431"    			// 数量
    ]
  ],
  "asks": [				// 卖单
    [
      "9638.2",			// 价格
      "12"				// 数量
    ]
  ]
}


'''