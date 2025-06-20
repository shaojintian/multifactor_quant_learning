import websocket
import json
import time

# WebSocket流地址
# <symbol>@depth 会每秒推送一次最新的全量深度
# <symbol>@depth@100ms 会每100ms推送一次增量更新
symbol = 'btcusdt'
stream_name = f"{symbol}@depth" # 每秒推送一次的增量更新
# stream_name = f"{symbol}@depth@100ms" # 每100ms推送一次的增量更新，数据量巨大！
ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"

def on_message(ws, message):
    """处理收到的消息"""
    data = json.loads(message)
    # 'e': event type, 'E': event time, 's': symbol
    # 'U': first update ID in event, 'u': final update ID in event
    # 'b': bids to be updated, 'a': asks to be updated
    
    # 在实际应用中，你需要在这里实现更新本地订单薄的逻辑
    # 1. 获取初始快照 (通过REST API)
    # 2. 缓存收到的websocket消息
    # 3. 对比消息的'U'和快照的'lastUpdateId'，应用缓存中的有效更新
    # 4. 持续应用新的更新
    
    print(f"Received update (First ID: {data['U']}, Final ID: {data['u']}):")
    # 只打印前几个更新以作演示
    print(f"  Bids updates: {data['b'][:1]}")
    print(f"  Asks updates: {data['a'][:1]}")
    print("-" * 20)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket closed ###")

def on_open(ws):
    print("### WebSocket opened ###")

if __name__ == "__main__":
    # websocket.enableTrace(True) # 开启调试信息
    ws = websocket.WebSocketApp(ws_url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    # 在一个单独的线程中运行WebSocket客户端
    ws.run_forever()