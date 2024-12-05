import pandas as pd
import numpy as np
import talib as ta
from scipy import stats
import yaml


def strength_pos_buy(z,n=None):
    '''主买主卖力量强弱对比'''
    yml_path = './set_up.yml'
    with open(yml_path,'rb')as fileOpen:
        # 通过safe_load 读取文件流
        value = yaml.safe_load(fileOpen)['DetailedSymbol']['contract']
    vwap = z['a']/z['v']/value  # t时刻的vwap
    midp = ((z['ap1'] + z['bp1']) / 2).shift(1)  # t-1时刻的mid price
    r = (vwap-midp)*2/(z['ap1']-z['bp1']) 
    return r.fillna(0)

def strength_pos_buy_version2(z,n):
    '''第二种定义主买的方法,计算主买的次数占滚动窗口中主买主卖和的比例'''
    yml_path = './set_up.yml'
    with open(yml_path,'rb')as fileOpen:
        # 通过safe_load 读取文件流
        value = yaml.safe_load(fileOpen)['DetailedSymbol']['contract']
    vwap = z['a']/z['v']/value  # t时刻的vwap
    midp = ((z['ap1'] + z['bp1']) / 2).shift(1)  # t-1时刻的mid price
    b,s = vwap > midp,vwap < midp
    f = ta.SUM(b,n)/(ta.SUM(b,n)+ta.SUM(s,n))
    return f 

def norm_active_amount(z, n):
    '''置信正态分布主动占比 '''
    fiducial_amount = z['a']*stats.norm.cdf(z['r']/0.1*1.96)
    return ta.SUM(fiducial_amount, n)/ta.SUM(z['a'], n)

def t_active_amount(z, n, d):
    '''t分布主动占比 '''
    # t为t分布的自由度，针对相同的股价变动，自由度越小，则根据t分布得到的主动买入金额占比越小。
    fiducial_amount = z['a']* t.cdf(z['r'], d)
    return ta.SUM(fiducial_amount, n)/ta.SUM(z['a'], n)

def s_big(z,n):
    '''来自于残差资金流强度因子构建，分子为大单买额-大单卖额,分母为abs(大单买额-大单卖额)'''
    # 先定义滚动窗口中的主买主卖，再定义大小单
    yml_path = './set_up.yml'
    with open(yml_path,'rb')as fileOpen:
        # 通过safe_load 读取文件流
        value = yaml.safe_load(fileOpen)['DetailedSymbol']['contract']
    vwap = z['a'] / z['v'] / value
    midp = ((z['ap1'] + z['bp1'])/2).shift(1)
    b,s = vwap>midp,vwap<midp  # 定义主买主卖
    big_order = z['a'] > (ta.MA(z['a'],120) + ta.STDDEV(z['a'],120))  # 定义大小单
    ba,sa = (b*big_order)*z['a'],(s*big_order)*z['a']
    return ta.SUM((ba-sa),n)/abs(ta.SUM((ba-sa),n))
    
def s_small(z,n):
    '''来自于残差资金流强度因子构建，分子为小单买额-小单卖额，分母为abs(小单买额-小单卖额)'''
    yml_path = './set_up.yml'
    with open(yml_path,'rb')as fileOpen:
        # 通过safe_load 读取文件流
        value = yaml.safe_load(fileOpen)['DetailedSymbol']['contract']
    vwap = z['a'] / z['v'] / value
    midp = ((z['ap1'] + z['bp1'])/2).shift(1)
    b,s = vwap>midp,vwap<midp  # 定义主买主卖
    small_order = z['a'] < (ta.MA(z['a'],120) - ta.STDDEV(z['a'],120))  # 定义大小单
    ba,sa = (b*small_order)*z['a'],(s*small_order)*z['a']
    return ta.SUM((ba-sa),n)/abs(ta.SUM((ba-sa),n))

# -------------------------------2 价格跳档--------------------------
def up_jump(z,n):
    '''价格上涨，跳档次数'''
    j = z['bp1'] > (z['ap1']).shift(1)
    return ta.SUM(j,n) / n

def dn_jump(z,n):
    '''价格下跌，跳档次数'''
    j = z['ap1'] < (z['bp1']).shift(1)
    return ta.SUM(j,n) / n

def net_jump(z,n):
    '''净跳档次数'''
    j1= z['bp1'] > (z['ap1']).shift(1)
    j2 = z['ap1'] < (z['bp1']).shift(1)
    return (ta.SUM(j1,n) - ta.SUM(j2,n)) / n

def vol_vol(z, n):
    '''高频成交量波动'''
    return ta.STDDEV(z['v'], n)/ta.MA(z['v'], n)

def jump_degree(z, n):
    '''跳跃度'''
    rtn = ta.ROCR(z['p'], 1)
    ln_rtn = ta.LN(ta.ROCR(z['p'], n))
    return (rtn - ln_rtn)*2 - ln_rtn**2

#-------------------------------4 流动性类---------------------------
def consequent_bid_ask_ratio(df, n):
    '''滚动窗口中有多少个bid和ask是连续一样的?计算他们的差值'''
    same_ask = ta.SUM(df['ap1'] == (df['ap1']).shift(n))
    same_bid = ta.SUM(df['bp1'] == (df['bp1']).shift(n))
    diff = same_ask - same_bid
    return diff.fillna(0)

def non_fluid_factor(z, n):
    '''经典非流动性因子,单位收益率由多少成交量推动'''
    epsilon = 1e-6
    ret = np.log(z.midp).diff()  # 对数收益率
    v = ta.SUM(z['v'], n) + 1e-6
    f = (ta.SUM(ret,n)/ta.SUM(v,n))*1e10  # TODO:为什么要乘以1e10
    return f  # 单位收益率由多少成交量推动？

def consequent_bid(z, n):
    '''滚动窗口中,有多少个bp1是一样的'''
    same_bid = z['bp1'] == z['bp1'].shift(1)
    return ta.SUM(same_bid, n) 

def consequent_ask(z, n):
    '''滚动窗口中,多少个ap1是一样的?'''
    same_ask = z['ap1'] == z['ap1'].shift(1)
    return ta.SUM(same_ask, n)

def consumption_rates(z, n):
    '''订单薄消耗速率'''
    vol_d = z['v']
    to_d = z['a']
    # 
    sv = (z['bp1'].shift(1) * vol_d - to_d) / (bidask_spread_shift := (z['bp1'] - z['ap1']).shift(1))
    bv = (to_d - (z['ap1'].shift(1)) * vol_d) / bidask_spread_shift

    sv[bv < 0] = vol_d[bv < 0]
    sv[sv < 0] = 0
    bv[bv < 0] = 0
    bv[sv < 0] = vol_d[sv < 0]

    return -ta.EMA(bv.ffill(), n) / z['bv1'] - ta.EMA(sv.ffill(), n) / z['av1']

def consequent_bid_ask_ratio(df, n):
    '''滚动窗口中有多少个bid和ask是连续一样的?计算他们的差值'''
    same_ask = ta.SUM(df['ap1'] == (df['ap1']).shift(n))
    same_bid = ta.SUM(df['bp1'] == (df['bp1']).shift(n))
    diff = same_ask - same_bid
    return diff.fillna(0)


# -------------------------------5 收益率相关----------------------------------
def up_ret_vol(z,n):
    '''高频上行波动占比'''
    mid = z[['ap1','bp1']].mean(1)
    yRtn = np.log(mid).diff()*1e4
    a = np.square(yRtn[yRtn>0])  # 计算一段时间窗口内yRtn大于0的平方和
    b = np.square(yRtn)  # 计算一段时间窗口内yRtn的平方和
    return ta.SUM(a,n)/ta.SUM(b,n)

def dn_ret_vol(z,n):
    '''高频下行波动占比'''
    mid = z[['ap1','bp1']].mean(1).shift(-1)
    yRtn = np.clip(np.log(mid).diff()*1e4,-5,5)
    a = np.square(yRtn[yRtn<0])  # 计算一段时间窗口内yRtn大于0的平方和
    b = np.square(yRtn)  # 计算一段时间窗口内yRtn的平方和
    return ta.SUM(a,n)/ta.SUM(b,n)

def im_up_dn(z,n):
    '''上下行波动率跳跃的不对称性'''
    mid = z[['ap1','bp1']].mean(1).shift(-1)
    yRtn = np.clip(np.log(mid).diff()*1e4,-5,5)
    u = np.square(yRtn[yRtn>0])
    d = np.square(yRtn[yRtn<0])
    return u-d

def ret_skew(z,n):
    '''高频已实现偏度，刻画股票日内快速拉升或下跌的特征，与收益率负相关'''
    mid = mid(z).shift(1)
    yRtn = np.square(np.log(mid).diff())
    return yRtn.rolling(n).skew()

def w_skewness(z, n):
    '''加权偏度'''
    def weighted_skewness(z):
        close_mean = z['c'].mean()
        vol_sum = z['v'].sum()
        weighted_skew = ((z['c'] - close_mean) /
                         close_mean)**3 * (z['v'] / vol_sum)
        return weighted_skew.sum()
    return z.rolling(window=n).apply(weighted_skewness)

def RV(z,n):
    '''已实现波动率,这里采用kaggle的波动率计算方式'''
    mid = mid(z)
    yRtn = np.log(mid).diff()
    rv = np.sqrt(ta.SUM(np.square(yRtn),n))
    return rv

    
# -------------------------------6 量价相关性----------------------------------
def corr_v_p(z,n):
    '''来自价量相关性平均数因子，计算价格和成交量的相关性系数'''
    return z['v'].rolling(n).corr(mid(z))

def corr_v_r_rate(z,n):
    '''成交量变化率和收益率的相关性''' 
    vr = z.v.diff()
    mid = z[['ap1','bp1']].mean(1)
    yRtn = np.log(mid).diff()
    return vr.rolling(n).corr(yRtn)

def acma(z, n):
    '''成交额自相关性'''
    return z['a'].rolling(window=n).corr((z['a']).shift(1))

# -------------------------------7 订单薄失衡----------------------------------

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

    return iof

    
# -------------------------------8 中间价变化率-------------------------------
def MPC(z,n):
    '''中间价的变化率'''
    epsilon = 1e3
    mid = (z[['ap1','bp1']].mean(1)) 
    m = ta.MOM(mid,1) + epsilon
    r = ta.ROC(m,n)
    f = r.replace([np.inf,-np.inf],0)
    return f

def MPC_skew(z,n):
    '''中间价变化率偏度'''
    epsilon = 1e3
    mid = (z[['ap1','bp1']].mean(1)) 
    m = ta.MOM(mid,1) + epsilon
    r = ta.ROC(m,1)
    r = r.rolling(n).skew()
    f = r.replace([np.inf,-np.inf],0)
    return f

# ------------------------tick因子测试的部分-------------------------
def tick_data_preprocess(z:pd.DataFrame) -> pd.DataFrame: 
    z['midp']=z[['ap1','bp1']].mean(1)
    _midp=z['midp'].shift(-1)
    y_cols=[]
    for n in range(0,61):
        y_cols.append(yn:=f'y{n*5}')
        if n==0:
            z[yn]=np.log(_midp).diff().shift(-2)*1e4
        else:
            y=np.log(_midp).diff(n*10).shift(-n*10-1)*1e4
            z[yn]=(y-y.mean())/y.std()
    return z

def ic(x:pd.Series, z:pd.DataFrame) -> None:
    '''ic decay for return'''
    ic_values = pd.Series({n*5:np.corrcoef(x.values,z[f'y{n*5}'].values)[0,1]   for n in range(1,61)})
    return ic_values

def bp_margin(x:pd.Series, z:pd.DataFrame) -> None:
    '''平均换手收益率'''
    pos_margin_values = pd.Series({n*5:((_:=ta.MA(x,n*10) if n else x)*z.y0).sum()/abs(_.diff()).sum() for n in range(1,61)})
    return pos_margin_values