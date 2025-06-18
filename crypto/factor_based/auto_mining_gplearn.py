import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from joblib import wrap_non_picklable_objects # 用于自定义函数
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

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
from util.sharpe_calculatio import cal_sharp_random,calculate_sharpe_ratio_corrected
from util import calculate_max_drawdown
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate,cal_net_values_compounded
from calculate_net_vaules import *
# from verify_risk_orthogonalization import risk_orthogonalization # 不再需要风险正交
pd.plotting.register_matplotlib_converters()
from factor_generator import *


# %%
# 0 data preprocess
_period_minutes = 60
_trading_hours = 24
# %%
#1. 读取行情数据
z = pd.read_csv(f'data/crypto/btcusdt_{_period_minutes}m.csv',index_col=0)
z.name = f"btcusdt_{_period_minutes}m"
import datetime
#date_threshold = datetime.datetime(2020, 2, 1)
#filtered_df = z[z.index > '2020-01-01']
filtered_df = z
filtered_df.index = pd.to_datetime(filtered_df.index, unit='ms', utc=True)
filtered_df = preprocess_data(filtered_df)
filtered_df = filtered_df.loc[filtered_df.index > pd.Timestamp("2020-06-01").tz_localize("UTC")]

# --- 0. 数据模拟 ---
# 在实际应用中，请替换成你自己的数据加载方式，例如 pd.read_csv('your_data.csv')
# 这里我们创建一个符合你列名的模拟数据集
columns = ['open', 'high', 'low', 'close', 'volume', 'close time',
           'quote asset volume', 'number of trades', 'taker buy base asset volume',
           'taker buy quote asset volume', 'ignore', 'log_return', 'volatility',
           'avg_volume']

# 创建1000条模拟的时间序列数据

df = filtered_df.copy()  # 使用 .copy() 来避免 SettingWithCopyWarning

print("模拟数据前5行:")
print(df.head())


# --- 1. 准备特征(X)和目标(y) ---
# ** 这是最关键的一步 **
# 我们的目标(y)是预测 *未来* 的收益率。这里我们用下一期的对数收益率作为目标。
# shift(-1) 表示将未来的值移动到当前行
df['target'] = df['log_return'].shift(-1)

# 删除最后一行，因为它的 'target' 是 NaN
df.dropna(inplace=True)

# 定义用作输入的特征列 (X)
# 我们排除掉未来信息('log_return', 'target')和非数值或不相关的列('close time', 'ignore')
feature_names = ['open', 'high', 'low', 'close', 'volume',
                 'quote asset volume', 'number of trades', 'taker buy base asset volume',
                 'taker buy quote asset volume', 'volatility', 'avg_volume']

X = df[feature_names].values
y = df['target'].values

# 对于时间序列数据，最好按时间顺序分割训练集和测试集
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]


# --- 2. 定义自定义函数 ---
# gplearn内置了一些基础函数，但对于金融数据，时序函数非常重要。
# 我们来定义几个 WorldQuant Alpha 101 中常见的函数

# 保护除法，避免除以0的错误
# 这个函数需要2个参数 (x1, x2)
def _protected_division(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, x1 / x2, 1.)

# 这个函数需要1个参数 (x)
def _ts_rank(x):
    window=10
    s = pd.Series(x)
    return s.rolling(window).apply(lambda w: w.rank(pct=True).iloc[-1], raw=False).fillna(0).values

# 这个函数需要1个参数 (x)
def _ts_delay(x):
    period  = 1
    s = pd.Series(x)
    return s.shift(period).fillna(0).values

# 这个函数需要2个参数 (x1, x2)
def _ts_corr(x1, x2):
    window=10
    s1 = pd.Series(x1)
    s2 = pd.Series(x2)
    return s1.rolling(window).corr(s2).fillna(0).values

# --- 包装函数，确保 arity 值完全正确 ---
# arity=2 用于需要两个输入的函数
protected_division = make_function(function=_protected_division, name='div', arity=2)
ts_corr = make_function(function=_ts_corr, name='ts_corr', arity=2)

# arity=1 用于需要一个输入的函数
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=1)
ts_delay = make_function(function=_ts_delay, name='ts_delay', arity=1)


# 创建函数集
user_function_set = [
    protected_division, 'add', 'sub', 'mul', 'neg', 'log', 'sqrt', 'abs',
    ts_rank, ts_delay, ts_corr
]


# --- 3. 配置并运行 SymbolicRegressor ---
print("\n开始进行遗传编程因子挖掘...")
est_gp = SymbolicRegressor(
    population_size=1000,         # 种群大小：每一代有多少个备选因子
    generations=20,               # 进化代数
    stopping_criteria=0.95,       # 停止标准：当适应度（metric）达到这个值时提前停止
    p_crossover=0.7,              # 交叉概率
    p_subtree_mutation=0.1,       # 子树变异概率
    p_hoist_mutation=0.05,        # 提升变异概率
    p_point_mutation=0.1,         # 点变异概率
    max_samples=0.9,              # 最大采样比例，用于随机子集训练，防止过拟合
    verbose=1,                    # 显示训练过程
    feature_names=feature_names,  # 特征名称，用于显示最终公式
    function_set=user_function_set, # 使用我们定义的函数集
    metric='pearson', # 优化目标：平均绝对误差。也可以是 'spearman' 或 'pearson'
    const_range=(-1., 1.),        # 公式中可以使用的常数范围
    random_state=42,              # 随机种子，保证结果可复现
    n_jobs=-1                     # 使用所有CPU核心并行计算
)

est_gp.fit(X_train, y_train)


# --- 以下是新增的代码，用于列出最好的10个因子 ---

print("\n" + "="*50)
print("--- 最好的10个因子 (按适应度排序) ---")
print("="*50)

# 1. 获取最终代的整个种群
# est_gp._programs 是一个包含所有最终个体的列表
final_population = est_gp._programs[-1]

# 2. 过滤掉适应度为无效值(None或nan)的个体，以防排序出错
valid_population = [
    p for p in final_population
]

# 3. 根据原始适应度(raw_fitness_)对种群进行排序
#    - gplearn 的默认适应度指标（如 'mean absolute error'）是越小越好。
#    - 如果你自定义的指标是越大越好, 请使用 reverse=True。
sorted_population = sorted(valid_population, key=lambda p: p.raw_fitness_)

# 4. 提取并打印前10个最好的因子
top_10_programs = sorted_population[:10]

if not top_10_programs:
    print("没有找到有效的因子。")
else:
    for i, program in enumerate(top_10_programs):
        print(f"\n排名 {i+1}:")
        print(f"  因子表达式: {program}")
        print(f"  适应度 (Fitness): {program.raw_fitness_:.6f}") # 格式化输出
        # 你也可以打印OOB适应度，它能更好地反映泛化能力
        if hasattr(program, 'oob_fitness_'):
             print(f"  OOB适应度 (OOB Fitness): {program.oob_fitness_:.6f}")
        print(f"  因子长度 (Length): {program.length_}")

# 你可以获取所有代的所有程序进行更深入的分析
# all_programs = est_gp._programs
# print(f"\n总共生成了 {len(all_programs)} 个程序。")