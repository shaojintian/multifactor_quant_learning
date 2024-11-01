# -*- coding: utf-8 -*-
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
from util.adjust_rep import adjust_positions
from datetime import datetime
from util.norm import normalize_factor
from util.sharpe_calculatio import cal_sharp_random
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate
from verify_risk_orthogonalization import process_multi_factors_nonlinear
pd.plotting.register_matplotlib_converters()



# %%
#1. 读取行情数据
z = pd.read_csv('data/510050.SH_15.csv',index_col=0)
import datetime
#date_threshold = datetime.datetime(2020, 2, 1)
#filtered_df = z[z.index > '2020-01-01']
filtered_df = z
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
# 6. 对应下一日行情收益率
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
ret.describe()

# # %%
# #### 3.进行风险正交
# from verify_risk_orthogonalization import risk_orthogonalization
# #factor = volatility_factor

# #单个因子的风险正交
# factors = pd.DataFrame({
#     'bolling_band_factor': bolling_band_factor,
#     'volatility_factor': volatility_factor,
#     'adaptive_momentum_factor': adaptive_momentum_factor
# })


#factor.hist().set_title(f"{factor.name}")
#normalized_factor.hist().set_title(f"{factor.name} normalized_factor")

# %%
# 4.处理因子
#print("处理前的因子统计："+f"{final_factor.shape}")
final_factor = volatility_factor
#print("处理后的因子统计："+f"{final_factor.shape} returns shape:{ret.shape}")
print(final_factor.describe())
final_factor.to_csv(f'factor_test_data/final_factor {final_factor.name} {datetime.datetime.now()} .csv')

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
#plt.hist(final_factor.dropna(), bins=50, alpha=0.3, label=final_factor.name)
#plt.show()

# %% 

# # %%
# # 7. 计算net_values_before_rebate
# from util.alignment import alignment
# final_factor, ret= alignment(final_factor,ret)

# %%
next_position = adjust_positions(final_factor,threshold = 0.5)
net_values = cal_net_values(next_position,ret)
plt.plot(net_values.values)
plt.title(final_factor.name)
plt.grid(True)
plt.show()


# %%
# define position ratio == normalized_factor

net_values_before_rebte = cal_net_values_before_rebate(next_position,ret)
plt.plot(net_values_before_rebte.values)
plt.title(final_factor.name+' before fee')
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
sharp = cal_sharp_random(cleaned_net_values,period_minutes=15,trading_hours=4)

print(f"Annualized Sharpe Ratio: {sharp:.4f}")


