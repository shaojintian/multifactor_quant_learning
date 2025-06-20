
# %%
# 在文件最开头添加以下代码
import os
import sys
# 正确的写法：
project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
sys.path.append(project_root)

import numpy as np
import pandas as pd
# import polars as pl
import matplotlib.pyplot as plt
from util.norm import normalize_factor
from util.sharpe_calculatio import *
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate,cal_net_values_compounded
from combine_factor import *
# from verify_risk_orthogonalization import risk_orthogonalization # 不再需要风险正交
pd.plotting.register_matplotlib_converters()
from binance.client import Client
from binance.enums import *
import json
from factor_generator import *
import logging
logger = logging.getLogger("sample")
logger.addHandler(logging.FileHandler("./crypto/logging.txt"))
logger.setLevel(logging.INFO)

# %%
# 0 data preprocess
_period_minutes = 60
_trading_hours = 24
_coin = "eth"

import websocket
import json
import pandas as pd
from datetime import datetime, timezone

csv_path = 'data/crypto/ethusdt_60m.csv'
coin = "eth"
period_minutes = 60

api_key = "pJAzMTnYORJU1ze6rDXkmR8RzDknYstsgbn9ZaHHdXw1cvcZEPjTLfPP0aGJnUFM"
api_secret = "rspRmZWi9WwBIu5EJXdtBWcwCCKzkSzk0rA44JxZVqsZrpou2TuLtBSXCfVsWZSu"

SYMBOL = "ETHUSDT"


def on_message(ws, message):
    data = json.loads(message)
    kline = data['k']
    #print(kline)

    if not kline['x']:  # 如果K线还没结束（即是实时更新中），忽略
        print("正在监听")
        return

    new_row = {
        'open': float(kline['o']),
        'high': float(kline['h']),
        'low': float(kline['l']),
        'close': float(kline['c']),
        'volume': float(kline['v']),
    }
    timestamp = int(kline['t'])

    # 加入到CSV文件
    
    filtered_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    #filtered_df.index = pd.to_datetime(filtered_df.index, unit='ms', utc=True)
    

    new_df = pd.DataFrame([new_row], index=[timestamp])
    df = pd.concat([filtered_df, new_df])
    df.to_csv(csv_path)
    #
    filtered_df = df.copy()
    filtered_df.index = pd.to_datetime(filtered_df.index, unit='ms', utc=True)
    df = preprocess_data(filtered_df)

    logger.info(f"[{timestamp}] New hourly bar added.")
    position = generate_signal(df)


    execute_trade(position)

def on_error(ws, error):
    logger.info("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed with code: {close_status_code}, message: {close_msg}")

def on_open(ws):
    print("WebSocket connection opened")

def run_websocket():
    stream = f"{coin}usdt@kline_1h"
    url = f"wss://stream.binance.com:9443/ws/{stream}"
    ws = websocket.WebSocketApp(url, on_message=on_message,
                                      on_error=on_error,
                                      on_close=on_close)
    ws.run_forever()


def generate_signal(df: pd.DataFrame):
    df = preprocess_data(df)

    # 获取最新一小时的数据（最后一行）
    filtered_df = df.copy()

    final_frame = add_factor(filtered_df, factor_logic_func=calculate_optimized_position_v2)
    final_frame = add_factor(final_frame, factor_logic_func=calculate_multi_period_momentum_filter_hourly)
    final_frame = add_factor(final_frame, factor_logic_func=greed_factor)
    final_frame = add_factor(final_frame, factor_logic_func=fct001)
    final_frame = add_factor(final_frame, factor_logic_func=calculate_ma)
    final_frame = add_factor(final_frame, factor_logic_func=calculate_momentum)
    final_frame = add_factor(final_frame, factor_logic_func=laziness_factor)
    final_frame = add_factor(final_frame, factor_logic_func=fear_factor)
    final_frame = add_factor(final_frame, factor_logic_func=fct007)
    final_frame = add_factor(final_frame, factor_logic_func=mean_revert_when_neutral_and_stable)
    final_frame = add_factor(final_frame, factor_logic_func=fct003)
    final_frame = add_factor(final_frame, factor_logic_func=fct004)
    final_frame = add_factor(final_frame, factor_logic_func=create_trend_following_vol_factor)
    final_frame = add_factor(final_frame, factor_logic_func=factor_bollinger_power)

    final_factor = combine_factors_lightgbm(
        final_frame,
        factor_cols=[
            "mean_revert_when_neutral_and_stable",
            "create_trend_following_vol_factor",
            "factor_bollinger_power",
            "calculate_ma",
            "fct001",
            "calculate_optimized_position_v2",
            "greed_factor",
            "calculate_multi_period_momentum_filter_hourly",
            "laziness_factor"
        ],
        weights=[0.359, 0.516833, 0.124167]
    )

    # 获取最新信号
    latest_signal = final_factor.iloc[-1]
    latest_signal_index = final_factor.index[-1]
    logger.info(f"📈 最新时间: {latest_signal_index}, 最新仓位: {latest_signal:.2f}")

    return latest_signal


def get_current_position(symbol="BTCUSDT"):
    pos = client.futures_position_information(symbol=symbol)[0]
    amt = float(pos["positionAmt"])
    price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
    balance = float(client.futures_account_balance()[0]["balance"])
    notional = amt * price
    return notional / balance if balance else 0.0

# 下市价单（按目标调整）
def place_order(side, notional, symbol):
    price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
    quantity = round(notional / price, 3)  # 精度视币种而定
    order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_BUY if side == "buy" else SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=quantity
    )
    logger.info(f"下单成功：{side} {quantity}张，notional={notional}")


def execute_trade(target_position):
    current = get_current_position(symbol=SYMBOL)
    delta = target_position - current
    logger.info(f"目标仓位: {target_position:.2f}，当前仓位: {current:.2f}，需要调整: {delta:.2f}")

    threshold = 0.2
    if abs(delta) < threshold:
        logger.info("调整幅度过小，跳过交易")
        return

    side = "buy" if delta > 0 else "sell"
    #place_order(side, abs(delta) * 100,SYMBOL)  # 100 是你账户的初始净值或资金基准


    # 可扩展推送逻辑，如：
    # send_signal_to_bot(latest_signal)
from binance.exceptions import BinanceAPIException
import ccxt
if __name__ == "__main__":
    binance = ccxt.binance({
    'apiKey': "pJAzMTnYORJU1ze6rDXkmR8RzDknYstsgbn9ZaHHdXw1cvcZEPjTLfPP0aGJnUFM",
    'secret': "rspRmZWi9WwBIu5EJXdtBWcwCCKzkSzk0rA44JxZVqsZrpou2TuLtBSXCfVsWZSu",
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # 表示使用 Binance Futures (USDT 永续)
    }
    })

    try:
        balance = binance.fetch_balance()
        print("连接成功 ✅")
        print("账户余额（USDT）:", balance['total']['USDT'])
    except ccxt.BaseError as e:
        raise(f"连接失败 {e}")
    
    print(get_current_position(symbol=SYMBOL))
    run_websocket()
