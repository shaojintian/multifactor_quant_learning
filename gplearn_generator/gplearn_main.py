import numpy as np
import pandas as pd
import time

from gplearn.genetic import SymbolicTransformer # 符号变换
from gplearn.fitness import make_fitness

from sklearn import metrics as me
import warnings
warnings.filterwarnings('ignore')


def preprocess(file_path, frequency, n_days):
    df = pd.read_csv(file_path)  # File should be in the form of Pickle

    t = 1
    df['etime'] = pd.to_datetime(df['open time'])
    df['ret'] = df['close'].pct_change(periods=t).shift(-t)
    # 注意这里return也需要给他做norm处理以保证数据无偏

    df.dropna(axis=0, how='any', inplace=True)
    df = df.reset_index(drop=True)

    fields = df.columns['open','high','low','close','volume','quote asset volume','number of trades','taker buy base asset volume','taker buy quote asset volume']  # e.g. ['etime', 'open', 'high', 'low', 'close', 'volume', 'ret'][1: ] # 特征

    for col in fields:
        df[col] = df[col].values.astype('float')
    
    X_train = df.drop(columns=['etime', 'ret']).to_numpy()
    y_train = df['ret'].values
    
    return np.nan_to_num(X_train), np.nan_to_num(y_train)


start_time = time.time()
X_train, y_train = preprocess(file_path='/Users/wanting/Downloads/multifactor_quant_learning/data/crypto/btcusdt_60m.csv', frequency='60', n_days=1)
# 不应该是全局数据，应该是70%，或者60%的数据，留一个验证集，测试集；

#"""Activate the Customized Functions"""

func_1 = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan']
#func_2 = [] # 这些都是在gplearn.functions里面，加入的自己的算子，比如说talib的，或者自己写的
func_2 = ['tanh', 'elu', 'TA_HT_TRENDLINE', 'TA_HT_DCPERIOD', 'TA_HT_DCPHASE', 'TA_SAR', 'TA_BOP', 'TA_AD', 'TA_OBV', 'TA_TRANGE'] # 算子

# 可以酌情把talib里面的算子全部加进去；

from gplearn.functions import _function_map
func_3 = list(_function_map.keys())

user_func = func_1 + func_2

# 给metric
def _my_metric(y, y_pred, w): # 打分机制--损失函数--作业2
    return me.normalized_mutual_info_score(y, y_pred) # 互信息

user_metric = make_fitness(function=_my_metric, greater_is_better=True, wrap=False)

# 以sharpe为基准的metric，请见SharpeMetric.py

# 最终生成的expression也是如此
# sin(relu(sigmoid(ta_cci(ta_kdj(ta_dma(close, high,30), ta_sma(close, 50))))))
# 统一化，标准化 (X - X.MEAN)/X.STD

ST_gplearn = SymbolicTransformer(population_size=500, # 一次生成因子的数量，
                                 hall_of_fame=100, # 
                                 n_components=100, # 最终输出多少个因子？
                                 generations=3, # 非常非常非常重要！！！--进化多少轮次？3也就顶天了，我一般上不超过2
                                 tournament_size=100,
                                 const_range=None, #(-1, 1),  # critical
                                 init_depth=(2, 5), # 第二重要的一个部位，控制我们公式的一个深度
                                #  function_set=user_func, # 输入的算子群
                                 function_set=user_func, # 输入的算子群
                                #  metric=user_metric, # 提升的点
                                 metric='pearson', # pearson相关系数：思考的深度不够，自己想办法，把sharpe写进来，本节课的终极作业；spearman
                                 parsimony_coefficient=0.005,
                                 p_crossover=0.4, 
                                 p_subtree_mutation=0.01,
                                 p_hoist_mutation=0.01,
                                 p_point_mutation=0.01,
                                 p_point_replace=0.4,
                                 feature_names=['open','high','low','close','volume','quote asset volume','number of trades','taker buy base asset volume','taker buy quote asset volume'], # 注意这里必须有feature_names 把重要算子“聚合化”，
                                 n_jobs=-1,
                                 random_state=1)

ST_gplearn.fit(X_train, y_train)
# 把这些expression，去给他通过正则表达式匹配，给他函数化。

#"""Rank and Save the Generated Alpha Factors"""


best_programs = ST_gplearn._best_programs # 取出最佳结果
best_programs_dict = {}


for bp in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(bp) + 1)
    best_programs_dict[factor_name] = {'fitness': bp.fitness_, 'expression': str(bp), 'depth': bp.depth_, 'length': bp.length_}

best_programs_frame = pd.DataFrame(best_programs_dict).T
best_programs_frame = best_programs_frame.sort_values(by='fitness', axis=0, ascending=False)
best_programs_frame = best_programs_frame.drop_duplicates(subset=['expression'], keep='first')

print(best_programs_frame)
best_programs_frame.to_csv('./fct_gp_1125.csv')

end_time = time.time()
print('Time Cost:-----    ', end_time-start_time, 'S    --------------')

# TA_HT_DCPERIOD(TA_OBV(TA_HT_DCPERIOD(tanh(low)), high))


