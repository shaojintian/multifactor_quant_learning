# 7. net value calculation 
# 1+position ratio * return - abs(position ratio change) * rebate
import numpy as np

def cal_net_values(pos: np.array, ret: np.array) -> np.array:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    fee = 0.0002  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = np.hstack((pos[0] - 0, np.diff(pos)))
    # 计算净值
    net_values = 1 + (pos * ret - np.abs(position_changes) * fee).cumsum()
    
    return net_values  # 返回净值序列

def cal_net_values_before_rebate(pos: np.array, ret: np.array) -> np.array:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    fee = 0.0002  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = np.hstack((pos[0] - 0, np.diff(pos)))
    # 计算净值
    net_values = 1 + (pos * ret).cumsum()
    #net_values = 1 + (pos * ret).cumsum()
    return net_values  # 返回净值序列