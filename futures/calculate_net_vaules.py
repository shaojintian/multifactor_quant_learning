# 7. net value calculation 
# 1+position ratio * return - abs(position ratio change) * rebate
import numpy as np
import pandas as pd

def cal_net_values(pos: pd.Series, ret: pd.Series) -> pd.Series:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    # np.savetxt('factor_test_data/crypto/pos.txt', pos, delimiter=',')
    # np.savetxt('factor_test_data/crypto/ret.txt', ret, delimiter=',')
    # if len(pos) != len(ret):
    #     raise ValueError("pos and ret must have the same length")
    #print(pos)
    fee = 0.0002  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = np.hstack((pos.iloc[0] - 0, np.diff(pos)))
    # 计算净值
    net_values = 1 + (pos * ret - np.abs(position_changes) * fee).cumsum()

    # error net vaule
    if net_values.iloc[-1] >100 :
        print('error net value')
        err_data = pd.DataFrame({'pos': pos, 'ret': ret})
        err_data.to_csv(f'factor_test_data/futures/error_net_values_{pos.name}.csv')
        return pd.Series(1, index=net_values.index)

    #fill 1
    net_values = net_values.dropna()
    return net_values  # 返回净值序列

def cal_net_values_before_rebate(pos: pd.Series, ret: pd.Series) -> pd.Series:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    fee = 0.0002  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = np.hstack((pos.iloc[0] - 0, np.diff(pos)))
    # 计算净值
    net_values = 1 + (pos * ret).cumsum()
    #net_values = 1 + (pos * ret).cumsum()

     #fill 1
    net_values = net_values.dropna()
    return net_values  # 返回净值序列