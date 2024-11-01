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
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()



# %%
#1. 读取行情数据
z = pd.read_csv('../data/510050.SH_15.csv',index_col=0)
import datetime
date_threshold = datetime.datetime(2020, 2, 1)
filtered_df = z[z.index > '2020-01-01']
#z.head()

# %%
from bolling_band_factor import bolling_band_factor_generator
from volatility_factor import calc_vol_mean_reversion_factor
from momentum_vol_factor import adaptive_momentum_factor
bolling_band_factor = bolling_band_factor_generator(filtered_df)
volatility_factor = calc_vol_mean_reversion_factor(filtered_df['close'])
adaptive_momentum_factor = adaptive_momentum_factor(filtered_df)


# %%
# v    
#volatility_factor
factors = [
    bolling_band_factor,
    volatility_factor,
    adaptive_momentum_factor
]

# %%
#### 3.进行风险正交
from verify_risk_orthogonalization import risk_orthogonalization
#factor = volatility_factor

#多个因子的风险正交
factors = pd.DataFrame({
    'bolling_band_factor': bolling_band_factor,
    'volatility_factor': volatility_factor,
    'adaptive_momentum_factor': adaptive_momentum_factor
})

# # 进行风险正交
# orthogonal_factors = risk_orthogonalization(factors)

# orthogonal_factors.describe()

#factor.hist().set_title(f"{factor.name}")
#normalized_factor.hist().set_title(f"{factor.name} normalized_factor")

# %%
from verify_risk_orthogonalization import process_multi_factors
# 4.处理因子
processed_factors, final_factor = process_multi_factors(factors)

# 检查处理后的分布
#print("处理后的因子统计：")
#print(processed_factors.describe())

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
for col in processed_factors.columns:
    plt.hist(processed_factors[col].dropna(), bins=50, alpha=0.3, label=col)
plt.legend()
plt.title("Processed Factors Distribution")
plt.show()

# %%

# %%
# 6. 计算行情收益率
ret = filtered_df['close'].shift(-1) / filtered_df['close'] - 1
ret.describe()

# %%
#ret.hist(bins=60)


# %%
# 7. net value calculation 
# 1+position ratio * return + abs(position ratio change) * rebate

def cal_net_values(pos: np.array, ret: np.array) -> np.array:
    '''计算净值序列
    pos: 仓位ratio[-300%,300%]
    ret: 未来1个周期的收益率
    '''
    fee = 0.0002  # 仓位每次变动的滑损(maker 0.02%， taker 0.05%)
    # 使用 np.hstack 组合当前仓位和仓位变化
    position_changes = np.hstack((pos[0] - 0, np.diff(pos)))
    # 计算净值
    net_values = 1 + (pos * ret - np.abs(position_changes * fee)).cumsum()
    #net_values = 1 + (pos * ret).cumsum()
    return net_values  # 返回净值序列


# %%
#factor.hist().set_title(f"{factor.name}")

# %%
# define position ratio == normalized_factor
pos = final_factor
net_values = cal_net_values(pos,ret)


plt.plot(net_values.values)
plt.title(final_factor.name)
plt.grid(True)
plt.show()


# %%
# 8. 计算annual夏普比率
cleaned_net_values = net_values[~np.isnan(net_values)]
sharp = cal_sharp_random(cleaned_net_values,period_minutes=15,trading_hours=4)

sharp




