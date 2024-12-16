# 山寨币轮动：当前1-3min成交额是之前序列时间窗口例如100个bar均值的50-100倍以上
# 本质是跟随知情交易者或操纵市场

import talib
import numpy as np

# 设置时间窗口大小和成交额倍数阈值
time_window = 100  # 100个bar
threshold = 100  # 成交额是均值的100倍

# 假设data是一个包含所有交易数据的DataFrame，包含时间、成交额等字段
def strategy(data):
    # 计算过去100个bar的成交额均值
    volumes = data['volume']  # 假设data中有成交量列
    mean_vol = talib.MA(volumes, time_window)
    
    # 当前的成交额
    current_vol = volumes[-1]
    
    # 检查当前成交额是否是均值的100倍以上
    if current_vol > mean_vol[-1] * threshold:
        return True  # 符合轮动条件
    else:
        return False  # 不符合轮动条件

# 需要在合适的时间点调用策略函数，例如每个3分钟bar结束时
# 假设data是每3分钟的成交数据，每次调用此策略来检测是否符合轮动条件
if strategy(data):
    print("符合轮动条件，执行策略")
else:
    print("不符合轮动条件")
