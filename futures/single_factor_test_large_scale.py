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
import talib as ta
import matplotlib.pyplot as plt
from util.norm import normalize_factor
from util.sharpe_calculatio import cal_sharp_random,cal_sharp
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate
from verify_risk_orthogonalization import process_multi_factors_nonlinear
pd.plotting.register_matplotlib_converters()

# %%
# 0 data preprocess
_period_minutes = 24 * 60
_trading_hours = 4

# %%
# 1. 读取行情数据
name = "csi500_futures_1d.csv"
z = pd.read_csv(os.path.join(project_root,f'data/commodities_data/{name}'), index_col=0)
z.name = name
filtered_df = z

#2.计算行情收益率
print(filtered_df['close'].describe())
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
ret.name = 'next returns'
print(ret.describe())

# %%
from factor_lib.bolling_band_factor import bolling_band_factor_generator
from factor_lib.volatility_factor import calc_vol_mean_reversion_factor
from factor_lib.momentum_vol_factor import adaptive_momentum_factor
from factor_lib.liquidity_factor import *
from factor_lib.alpha101 import get_alpha
from factor_benchmark import permutation_test

# 生成因子
bolling_band_factor = bolling_band_factor_generator(filtered_df)
volatility_factor = calc_vol_mean_reversion_factor(filtered_df['close'])
adaptive_momentum_factor = adaptive_momentum_factor(filtered_df)
normalized_volatility_adjusted_momentum = volatility_adjusted_momentum(filtered_df)
normalized_volume_weighted_momentum = volume_weighted_momentum(filtered_df)
normalized_buy_pressure = buy_pressure(filtered_df)
normalized_price_efficiency = price_efficiency(filtered_df)
normalized_price_volume_divergence = price_volume_divergence(filtered_df)
normalized_volatility_regime = volatility_regime(filtered_df)
normalized_trade_activity = trade_activity(filtered_df)
normalized_price_strength = price_strength(filtered_df)
normalized_volume_imbalance = volume_imbalance(filtered_df)
normalized_multi_period_momentum = multi_period_momentum(filtered_df)
alphas = get_alpha(filtered_df)

# %%

# %%
# 定义单因子测试函数
def test_single_factor(factor_name, factor_data):
    # 将因子放入DataFrame
    factors = pd.DataFrame({factor_name: factor_data})

    # 处理因子
    print(f"处理因子: {factor_name}")
    #processed_factors, final_factor = process_multi_factors_nonlinear(factors, returns=ret)
    final_factor = factor_data.fillna(0)
    final_factor.to_csv(f'reports/{z.name}_{factor_name}_factor.csv')
    # 计算净值
    net_values = cal_net_values(final_factor, ret)
    

    #保存净值，脚本画图
    net_values.to_csv(f'reports/{z.name}_{factor_name}_net_values.csv')

    # plt.figure(figsize=(12, 6))
    # plt.plot(net_values.index, net_values.values)
    # plt.title(f'Net Value of {factor_name}')
    # plt.xlabel('Time (UTC)')
    # plt.ylabel('net value')
    # plt.legend()
    # plt.grid()
    # plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好地显示
    # plt.tight_layout()  # 自动调整布局
    # plt.show()
    
    # 计算夏普比率
    cleaned_net_values = net_values[~np.isnan(net_values)]
    sharp = cal_sharp(cleaned_net_values)
    
    # 计算p值
    p_value = permutation_test(final_factor, ret, n_permutations=1000)
    # 可视化
    # plt.hist(final_factor.dropna(), bins=50, alpha=0.3, label=final_factor.name)
    # plt.title(f"Histogram of {factor_name}")
    # plt.show()
    
    return sharp, p_value

# %%
# 对每个因子进行测试
factors = {
    # 'bolling_band_factor': bolling_band_factor,
    # 'volatility_factor': volatility_factor,
    # 'adaptive_momentum_factor': adaptive_momentum_factor,
    # 'normalized_volatility_adjusted_momentum': normalized_volatility_adjusted_momentum,
    # 'normalized_volume_weighted_momentum': normalized_volume_weighted_momentum,
    # 'normalized_buy_pressure': normalized_buy_pressure,
    # 'normalized_price_efficiency': normalized_price_efficiency,
    # 'normalized_price_volume_divergence': normalized_price_volume_divergence,
    # 'normalized_volatility_regime': normalized_volatility_regime,
    # 'normalized_trade_activity': normalized_trade_activity,
    # 'normalized_price_strength': normalized_price_strength,
    # 'normalized_volume_imbalance': normalized_volume_imbalance,
    # 'normalized_multi_period_momentum': normalized_multi_period_momentum,
    'alpha005': normalize_factor(alphas['alpha005']),
    'alpha005_clip03': normalize_factor(alphas['alpha005']).clip(0,3),
    'alpha005_clip13': normalize_factor(alphas['alpha005']).clip(1,3),
    # 'alpha037': normalize_factor(alphas['alpha037']),
    # 'alpha038': normalize_factor(alphas['alpha038']),
    # 'alpha049': normalize_factor(alphas['alpha049']),
    # 'alpha041': normalize_factor(alphas['alpha041']),
    # 'alpha025': normalize_factor(alphas['alpha025']), 
}
# alpha_dict = {}
# for i in range(1, 101):
#     try:
#         alpha_dict[f'alpha{i:03}'] = normalize_factor(alphas[f'alpha{i:03}']) 
#     except:
#         alpha_dict[f'alpha{i:03}'] = pd.Series() # Accessing the alphas dictionary
# factors.update(alpha_dict)


# %%
# 创建一个空的 DataFrame 来存储结果
results = []
for name, data in factors.items():
    if data.empty:
        continue
    sharp_ratio, p_value = test_single_factor(name, data)
    # 将结果添加到列表中
    results.append({'因子': name, '年化夏普比率': sharp_ratio, 'p值': p_value})
# 将结果转换为 DataFrame
results_df = pd.DataFrame(results)

# 保存为 CSV 文件
from datetime import datetime
results_df.to_csv(f'reports/{z.name}_{datetime.now()}_factor_results.csv', index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 编码以支持中文字符
print("report saved=====================\n")

# %% save final_factor
# 这里可以选择是否保存最终因子
# final_factor.to_csv('factor_test_data/crypto/final_factor.csv')
# ret.to_csv('factor_test_data/crypto/ret.csv')

# 其他可视化和分析代码可以根据需要添加