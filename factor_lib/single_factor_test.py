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
from util.sharpe_calculatio import cal_sharp_random
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate
from verify_risk_orthogonalization import process_multi_factors_nonlinear
pd.plotting.register_matplotlib_converters()

# %%
# 0 data preprocess
_period_minutes = 60
_trading_hours = 24

# %%
# 1. 读取行情数据
z = pd.read_csv(f'data/crypto/btcusdt_{_period_minutes}m.csv', index_col=0)
z.name = f"btcusdt_{_period_minutes}m"
filtered_df = z

# %%
from bolling_band_factor import bolling_band_factor_generator
from volatility_factor import calc_vol_mean_reversion_factor
from momentum_vol_factor import adaptive_momentum_factor

# 生成因子
bolling_band_factor = bolling_band_factor_generator(filtered_df)
volatility_factor = calc_vol_mean_reversion_factor(filtered_df['close'])
adaptive_momentum_factor = adaptive_momentum_factor(filtered_df)

# %%
# 6. 计算行情收益率
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
print(ret.describe())

# %%
# 定义单因子测试函数
def test_single_factor(factor_name, factor_data):
    # 将因子放入DataFrame
    factors = pd.DataFrame({factor_name: factor_data})

    # 处理因子
    print(f"处理因子: {factor_name}")
    processed_factors, final_factor = process_multi_factors_nonlinear(factors, returns=ret)
    
    # 计算净值
    net_values = cal_net_values(final_factor, ret)
    
    # 计算夏普比率
    cleaned_net_values = net_values[~np.isnan(net_values)]
    sharp = cal_sharp_random(cleaned_net_values, period_minutes=_period_minutes, trading_hours=_trading_hours)
    
    # 可视化
    plt.hist(final_factor.dropna(), bins=50, alpha=0.3, label=final_factor.name)
    plt.title(f"Histogram of {factor_name}")
    #plt.show()
    
    return sharp, final_factor.describe()

# %%
# 对每个因子进行测试
factors = {
    'bolling_band_factor': bolling_band_factor,
    'volatility_factor': volatility_factor,
    'adaptive_momentum_factor': adaptive_momentum_factor
}

# 打开文件以写入结果
with open(f'reports/{z.name}_factor_results.txt', 'w') as f:
    for name, data in factors.items():
        sharp_ratio, description = test_single_factor(name, data)
        # 将结果写入文件
        f.write(f"因子 {name} 的年化夏普比率: {sharp_ratio:.4f}\n")
        #f.write(f"因子 {name} 的描述统计: \n{description}\n\n")

# %% save final_factor
# 这里可以选择是否保存最终因子
# final_factor.to_csv('factor_test_data/crypto/final_factor.csv')
# ret.to_csv('factor_test_data/crypto/ret.csv')

# 其他可视化和分析代码可以根据需要添加