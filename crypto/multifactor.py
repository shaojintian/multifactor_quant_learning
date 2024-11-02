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
project_root = os.path.abspath("/Users/wanting/Downloads/multifactor_quant_learning")
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
#1. 读取行情数据
z = pd.read_csv(os.path.join(project_root, f'data/crypto/btcusdt_{_period_minutes}m.csv'),index_col=0)
z.name = f"btcusdt_{_period_minutes}m"
import datetime
#date_threshold = datetime.datetime(2020, 2, 1)
#filtered_df = z[z.index > '2020-01-01']
filtered_df = z
#z.head()

# %%
from factor_lib.bolling_band_factor import bolling_band_factor_generator
from factor_lib.volatility_factor import calc_vol_mean_reversion_factor
from factor_lib.momentum_vol_factor import adaptive_momentum_factor
from factor_lib.liquidity_factor import *
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




# %%
# 6. 计算行情收益率
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
print(ret.describe())

#factor = volatility_factor

#多个因子的风险正交
factors = pd.DataFrame({
    'bolling_band_factor': bolling_band_factor,
    # # 'volatility_factor': volatility_factor,
    # # 'adaptive_momentum_factor': adaptive_momentum_factor,
    # # 'normalized_volatility_adjusted_momentum': normalized_volatility_adjusted_momentum,
    # # 'normalized_volume_weighted_momentum': normalized_volume_weighted_momentum,
    # 'normalized_buy_pressure': normalized_buy_pressure,
    # # 'normalized_price_efficiency': normalized_price_efficiency,
    # # 'normalized_price_volume_divergence': normalized_price_volume_divergence,
    # 'normalized_volatility_regime': normalized_volatility_regime,
    # # 'normalized_trade_activity': normalized_trade_activity,
    # # 'normalized_price_strength': normalized_price_strength,
    # 'normalized_volume_imbalance': normalized_volume_imbalance,
    # # 'normalized_multi_period_momentum': normalized_multi_period_momentum
})


#factor.hist().set_title(f"{factor.name}")
#normalized_factor.hist().set_title(f"{factor.name} normalized_factor")

# %%
# 4.处理因子
print("4.处理前的因子统计："+f"{factors.describe()}")
processed_factors, final_factor = process_multi_factors_nonlinear(factors, returns=ret)
print(f"/n=====处理后的最终因子统计： {final_factor.shape}")
print(final_factor.describe())

# 检查处理后的分布
#print("处理后的因子统计：")
#print(processed_factors.describe())

# # 可视化
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# for col in processed_factors.columns:
#     plt.hist(processed_factors[col].dropna(), bins=50, alpha=0.3, label=col)
# plt.legend()
# plt.title("Processed Factors Distribution")
# #plt.show()

# %% 
plt.hist(final_factor.dropna(), bins=50, alpha=0.3, label=final_factor.name)
plt.show()


# %% 因子转化为仓位,
# 
final_factor = final_factor.fillna(0) 

# %% save final_factor
final_factor.to_csv(os.path.join(project_root,'factor_test_data/crypto/final_factor.csv'))
ret.to_csv(os.path.join(project_root,'factor_test_data/crypto/ret.csv'))

# %%  #
# cal   calculate 净值

net_values = cal_net_values(final_factor,ret)
plt.plot(net_values.values)
plt.title(f"{z.name} "+ final_factor.name)
plt.grid(True)
plt.show()



# %%
# calculate 净值 before fee
net_values_before_rebte = cal_net_values_before_rebate(final_factor,ret)
plt.plot(net_values_before_rebte.values)
plt.title(f"{z.name} "+final_factor.name+'before fee')
plt.grid(True)
plt.show()

# %%
# 可视化组合因子的预测效果
plt.figure(figsize=(12, 6))
plt.scatter(final_factor, ret, alpha=0.5)
plt.xlabel('Combined Factor Prediction')
plt.ylabel('Actual Returns')
plt.title('Combined Factor vs Actual Returns')
plt.grid(True)
plt.show()



# %%
# 8. 计算annual夏普比率
cleaned_net_values = net_values[~np.isnan(net_values)]
sharp = cal_sharp_random(cleaned_net_values,period_minutes=_period_minutes,trading_hours=_trading_hours)

print(f"Annualized Sharpe Ratio: {sharp:.4f}")


