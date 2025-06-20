import pandas as pd
import numpy as np
import talib as ta

# ------------------------订单薄失衡--------------------------
def ofi(z:pd.DataFrame,n:int = 3) -> pd.Series:
    '''订单薄失衡'''
    # 计算bid一侧
    bid_p_previous = z[f'bp1'].shift(1)
    bid_p_current = z[f'bp1']
    # 当前bid price大于上一刻的bid price,增量为当前的挂单量也就是bid_v
    delta_v1 = (bid_p_current > bid_p_previous) * z['bv1']
    # 当前bid price小于上一刻的bid price,增量为前一刻被成交的量取负数
    delta_v2 = (bid_p_current < bid_p_previous) * z['bv1'].shift(1) * -1.
    # 当前bid price等于上一刻的bid price,增量为当前的挂单量减去前一刻的挂单量
    delta_v3 = (bid_p_current == bid_p_previous) * (z['bv1'] - z['bv1'].shift(1))
    # 三者相加，得到最终的delta_v
    delta_bid_v = delta_v1 + delta_v2 + delta_v3

    # 计算ask一侧
    ask_p_previous = z['ap1'].shift(1)
    ask_p_current = z['ap1']
    # 当前ask price大于上一刻的ask price,增量为前一刻被成交量取负数
    delta_v1 = (ask_p_current > ask_p_previous) * z['av1'].shift(1) * -1.
    # 当前aid price小于上一刻的aid price,增量为当前的挂单量也就是bid_v
    delta_v2 = (ask_p_current < ask_p_previous) * z['av1'].shift(1) * -1.
    # 当前ask price等于上一刻的ask price,增量为当前的挂单量减去前一刻的挂单量
    delta_v3 = (ask_p_current == ask_p_previous) * (z['av1'] - z['av1'].shift(1))
    # 三者相加，得到最终的delta_v
    delta_ask_v = delta_v1 + delta_v2 + delta_v3

    iof = delta_bid_v - delta_ask_v

    return iof.fillna(0)

# ---------------------------主动大单成交-------------------------------
# 向量形式写的
def positive_buy_1(z:pd.DataFrame,n:int) -> pd.Series:
    '''成交均价大于上一时刻的mid price'''
    # 计算vwap要除以合约乘数 焦煤 60
    vwap = z['a'] / z['v'] / 60
    # 在快照数据上刻画主动成交，比上一刻mid price((bp1+ap1)/2)，就可以定义成主买
    pos_buy = vwap>z['mid'].shift(1) 
    return ta.SUM(pos_buy,n)

def positive_buy_2(z:pd.DataFrame,n:int) -> pd.Series:
    '''bp1大于上一时刻的ap1'''
    # 买一的价格上移，超过了之前的卖一价格，可以定义成主买
    pos_buy = z['bp1'] > z['ap1'].shift(1) 
    return ta.SUM(pos_buy,n)

def big_order(z:pd.DataFrame,n:int) -> pd.Series:
    '''大单成交'''
    # 成交额大于2倍标准差，定义为大单
    big_order = z['a'] > (ta.MA(z['a'],120) + 2*ta.STDDEV(z['a'],120)) # 1 0 
    return ta.SUM(big_order,n)

def big_order_strength(z,n):
    '''大单买入强度'''
    # 1 定义大买单
    big_order = z['a'] > (ta.MA(z['a'],120) + 2*ta.STDDEV(z['a'],120)) 
    # 2 定义大买单成交额
    big_order_amount = big_order * z['a']
    # 3 定义大买单成交额的均值
    mean_big_order_amount = ta.MA(big_order_amount,n)
    # 4 定义大买单成交额的标准差
    std_big_order_amount = ta.STDDEV(big_order_amount,n)

    return mean_big_order_amount/std_big_order_amount

# 还可以继续做衍生，比如主动成交的大单金额，占滚动窗口总金额的比重？主动成交大单金额的其他统计特征
# 除法/减法，体现了一种对比
# 一种基础的特征加上多种的衍生方式，统计特征，最大值/最小值/均值/标准差/偏度/峰度
# micro pice weighted mid price

# ----------------------------量与价的相关性---------------------------
def rtn_vol(z:pd.DataFrame, n:int) -> pd.Series:
    '''绝对收益率和成交量相关性'''
    abs_rtn = np.log(z['mid']).diff()
    return abs_rtn.rolling(window=n).corr(z['v'])

def amount_corr(z:pd.DataFrame, n:int) -> pd.Series:
    '''成交额自相关性'''
    # ta.CORREAL
    return z['a'].rolling(window=n).corr((z['a'].shift(1)))

# --------------------------波动率相关---------------------------
def up_rtn(z:pd.DataFrame,n:int) -> pd.Series:
    '''已实现上行波动率'''
    rtn = np.log(z['mid']).diff()
    up = rtn > 0
    up_rtn = rtn * up
    return ta.SUM(up_rtn,n)

# ---------------------------计算因子的ic decay------------------------
def tick_data_preprocess(z:pd.DataFrame) -> pd.DataFrame: 
    z['midp']=z[['ap1','bp1']].mean(1)  # 算出当前时刻的中间价
    for n in range(0,61):
        if n==0:
            z[yn]=np.log(z.midp).diff().shift(-1)*1e4  # 计算对数收益率，注意tick交易的执行延时
        else:
            y=np.log(z.midp).diff(n*10).shift(-n*10-1)*1e4
            z[yn]=(y-y.mean())/y.std()  # 为什么要除以std？是因为要滤除长周期的趋势，做到波动率归一化
    return z

def ic(x:pd.Series, z:pd.DataFrame) -> None:
    '''ic decay for return'''
    ic_values = pd.Series({n*5:np.corrcoef(x.values,z[f'y{n*5}'].values)[0,1]  for n in range(1,61)})
    return ic_values


if __name__ == "__main__":
    # --- 模拟生成一份微观数据 ---
    # 时间跨度为几个小时，频率为亚秒级
    time_index = pd.to_datetime(pd.date_range(start='2023-10-27 09:00:00', end='2023-10-27 12:00:00', freq='200L')) # 200毫秒
    df_micro = pd.DataFrame(index=time_index)
    df_micro.index.name = 'timestamp'

    # 交易数据
    df_micro['price'] = 29000 + np.random.randn(len(df_micro)).cumsum() * 0.1
    df_micro['volume'] = np.random.rand(len(df_micro)) * 0.5 + 0.01
    df_micro['side'] = np.random.choice(['buy', 'sell'], size=len(df_micro), p=[0.55, 0.45])

    # 订单薄数据
    spread = 0.1
    df_micro['bp1'] = df_micro['price']
    df_micro['ap1'] = df_micro['price'] + spread
    df_micro['bv1'] = np.random.uniform(5, 15, size=len(df_micro))
    df_micro['av1'] = np.random.uniform(5, 15, size=len(df_micro))
    df_micro['spread'] = df_micro['ap1'] - df_micro['bp1']

    # 假设OFI已经计算好 (在实际应用中，你需要先调用ofi函数)
    df_micro['ofi'] = ofi(df_micro)

    print("原始微观数据 (前几行):")
    print(df_micro.head())

