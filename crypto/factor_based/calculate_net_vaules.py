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
    #     raise ValueError("pos and ret must have the same length",len(pos),len(ret))
    #print(pos)
    fee = 0.0005  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)

    common_index = pos.index.intersection(ret.index)
    pos, ret = pos.loc[common_index], ret.loc[common_index]
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = pos.diff().fillna(0)

    # 小于手续费的不执行交易，手续费为 0 仓位变化大于50%
    should_trade = np.abs(position_changes) > 0.5  # boolean mask 0.5
    # 实际持仓：不交易时，仓位用前一时刻仓位；交易时用当前仓位
    effective_pos = pos.copy()
    effective_pos = effective_pos.where(should_trade, other=pos.shift(1))

    #print("--effective_pos--", effective_pos.describe())

    # 手续费，只有交易时收
    effective_fee = np.where(should_trade, np.abs(position_changes) * fee, 0)

    # 净值计算
    net_values = 1 + (effective_pos * ret - effective_fee).cumsum()

    #fill 1
    net_values = net_values.dropna()
    return net_values  # 返回净值序列



    
def cal_net_values_soft_stop(pos: pd.Series, ret: pd.Series, fee: float = 0.0005, stop_drawdown: float = 0.08, resume_drawdown: float = 0.04) -> pd.Series:
    """
    计算净值序列，加入最大回撤软止损逻辑：
    - 回撤超过8%时清仓
    - 回撤缩小到4%以内或净值创新高时恢复交易
    """
    pos = pos.copy()
    position_changes = pos.diff().fillna(0)
    should_trade = np.abs(position_changes) > 0.5
    effective_pos = pos.where(should_trade, other=pos.shift(1)).fillna(0)

    net_values = [1.0]
    max_net = 1.0
    stop = False
    current_pos = 0

    for i in range(len(ret)):
        fee_cost = abs(position_changes.iloc[i]) * fee if should_trade.iloc[i] else 0

        # 回撤控制
        if stop:
            current_pos = 0  # 清仓
        else:
            current_pos = effective_pos.iloc[i]

        # 本期净值
        pnl = current_pos * ret.iloc[i] - fee_cost
        new_net = net_values[-1] + pnl
        net_values.append(new_net)

        # 更新最大净值
        if new_net > max_net:
            max_net = new_net
            stop = False  # 创新高 → 恢复交易

        # 计算回撤
        dd = (new_net - max_net) / max_net

        # 触发止损
        if dd < -stop_drawdown:
            stop = True  # 进入止损状态

        # 若回撤恢复到阈值之上 → 恢复交易
        elif stop and dd > -resume_drawdown:
            stop = False

    net_values = pd.Series(net_values[1:], index=ret.index)
    return net_values

    



def cal_net_values_before_rebate(pos: pd.Series, ret: pd.Series) -> pd.Series:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    fee = 0.0005  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    # 小于手续费的不执行交易，手续费为 0 仓位变化大于50%才交易
    common_index = pos.index.intersection(ret.index)
    pos, ret = pos.loc[common_index], ret.loc[common_index]

    position_changes = np.hstack((pos.iloc[0] - 0, np.diff(pos)))
    should_trade = np.abs(position_changes) > 1000 * fee  # boolean mask 0.5
    
    effective_pos = pos.copy()
    effective_pos = effective_pos.where(should_trade, other=pos.shift(1))
    # 计算净值
    net_values = 1 + (effective_pos * ret ).cumsum()

    #fill 1
    net_values = net_values.dropna()
    return net_values  # 返回净值序列
def cal_net_values_compounded(pos: pd.Series, ret: pd.Series, fee: float = 0.0005, initial_value: float = 1.0) -> pd.Series:
    """
    计算复利净值序列（使用未来收益率作为输入）。

    该函数使用标准的乘法模型（复利）进行回测，更符合真实世界的投资情况。
    它遵循了“先交易付费，后持仓获利”的逻辑顺序。

    Args:
        pos (pd.Series): 仓位序列。pos[t] 是为 [t, t+1] 周期设定的目标仓位。
        ret (pd.Series): 未来1个周期的收益率。ret[t] 是资产在 [t, t+1] 期间的收益率。
        fee (float, optional): 单边交易费率。每次仓位调整，按调整部分比例收费。默认为 0.0005。
        initial_value (float, optional): 初始净值。默认为 1.0。

    Returns:
        pd.Series: 复利计算后的净值序列。
    """
    # --- 输入验证 ---
    if len(pos) != len(ret):
        raise ValueError("pos and ret must have the same length")
    if not pos.index.equals(ret.index):
        # 索引对齐非常重要，可以避免很多难以察觉的错误
        print("Warning: Indexes of pos and ret are not aligned. Forcing alignment.")
        pos, ret = pos.align(ret, join='inner')

    # --- 核心计算 ---

    # 1. 计算仓位变化
    # 使用 shift(1) 获取上一期的仓位，fillna(0) 处理第一个时间点的特殊情况（从0仓位开始）
    prev_pos = pos.shift(1).fillna(0)
    position_changes = pos - prev_pos

    should_trade = np.abs(position_changes) > 0.5  # boolean mask 0.5
    
    effective_pos = pos.copy()
    effective_pos = effective_pos.where(should_trade, other=pos.shift(1))
    effective_fee = np.where(should_trade, np.abs(position_changes) * fee, 0)

    # 2. 计算每个周期的净值乘数
    # 乘数 = (1 - 交易成本率) * (1 + 持仓收益率)
    
    # 交易成本会立即减少我们的本金
    fee_multiplier = 1 - effective_fee
    
    # 持仓收益作用于交易后剩下的本金上
    pnl_multiplier = 1 + effective_pos * ret
    
    # 单周期的总回报乘数
    daily_multiplier = fee_multiplier * pnl_multiplier

    # 3. 计算复利净值
    # 通过累乘（cumprod）计算净值曲线
    net_values = daily_multiplier.cumprod() * initial_value
    
    # 如果原始序列的第一个值是 NaN，cumprod 结果也会是 NaN，这里我们用初始值填充
    # 但根据我们的逻辑，第一个值已经计算过了，所以通常不需要这步
    # 不过为了稳健性可以保留
    net_values = net_values.fillna(initial_value)

    return net_values


def cal_turnover_annual(pos: pd.Series, periods_per_year: int = 365) -> float:
    """
    计算年化换手率（仓位变动率总和年化）。

    Args:
        pos (pd.Series): 仓位序列，索引应为时间序列。
        periods_per_year (int): 每年周期数，默认252个交易日。

    Returns:
        float: 年化换手率（总仓位变动绝对值年化）
    """
    fee = 0.0005  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    raw_position_changes = pos.diff().fillna(0)

    # 小于手续费的不执行交易，手续费为 0 仓位变化大于50%
    should_trade = np.abs(raw_position_changes) > 10000000000 * fee  # boolean mask 0.5
    # 实际持仓：不交易时，仓位用前一时刻仓位；交易时用当前仓位
    effective_pos = pos.copy()
    effective_pos = effective_pos.where(should_trade, other=pos.shift(1))

    position_changes = effective_pos.diff().fillna(effective_pos.iloc[0]).abs()
    # 日均换手率
    daily_turnover = position_changes.sum() / len(pos)
    # 年化换手率
    annual_turnover = daily_turnover * periods_per_year
    return annual_turnover


