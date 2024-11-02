# %%
# 置换检验
import numpy as np 
import pandas as pd
from scipy.stats import pearsonr

def permutation_test(x, y, n_permutations=100) -> float:
    """执行置换检验以评估x和y之间的关系是否显著。
    -----------------------------------------
    Params:
    x (ndarray): original factor Series
    y (ndarray): Future return series
    n_permutations (int): numbers of permutation
    -----------------------------------------
    Return:
    p_value (float): p_value,the smaller,the better
    -----------------------------------------"""

    original_corr = pearsonr(x, y)[0]  # 计算原始数据的相关系数

    permuted_x = np.empty((n_permutations, len(x)))  # 创建所有置换的矩阵
    for i in range(n_permutations):
        permuted_x[i] = np.random.permutation(x)

    # 计算所有置换的相关系数
    permuted_corrs = np.array([pearsonr(permuted_x[i], y)[0] for i in range(n_permutations)])
    p_value = np.mean(np.abs(permuted_corrs) >= np.abs(original_corr))  # 我们期待p值越小越好
    return p_value

# %%
# ic和sharp
def cal_ic(x,y):
    '''计算ic'''
    return np.corrcoef(x,y)[0,1]
def cal_sharp(x,y):
    '''计算Sharp,x为仓位,y为未来一个周期收益率'''
    trading_days = 252
    net_values = 1+(x*y).cumsum()
    returns = (net_values[1:]-net_values[:-1])/net_values[:-1]
    ret_mean = returns.mean()
    ret_std = returns.std()
    risk_free = 0.03
    sharp = (ret_mean*trading_days-risk_free)/ret_std*np.sqrt(trading_days)
    return sharp

# %%
# 一道面试题，怎么计算滚动的ic？
import talib as ta
def rolling_ic_1(x,y):
    '''x、y是ndarray数组'''
    return ta.CORREL(x,y,10)
def rolling_ic_2(x,y):
    '''x、y是pdseries'''
    return x.rolling(10).corr(y)


# %%
# 因子的自相关性检验
# 设置窗口大小，例如窗口大小为3
def rolling_autoic(x):
    window_size = 10
    rolling_autocorrelation = x.rolling(window=window_size).apply(lambda x: x.autocorr(lag=1))
    return rolling_autocorrelation


