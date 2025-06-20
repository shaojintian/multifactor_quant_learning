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
from util.sharpe_calculatio import cal_sharp_random,calculate_sharpe_ratio_corrected,calculate_calmar_ratio
from util import calculate_max_drawdown
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate,cal_net_values_compounded
from calculate_net_vaules import *
# from verify_risk_orthogonalization import risk_orthogonalization # 不再需要风险正交
pd.plotting.register_matplotlib_converters()
from factor_generator import *


# %%
# 0 data preprocess
_cal_peroid = 5
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
#filtered_df = filtered_df.loc[filtered_df.index > pd.Timestamp("2024-06-01").tz_localize("UTC")]

# z.head()  # 注释掉以避免执行

# %%
#2. 生成各种备选因子
zz = pd.read_csv(f'data/crypto/btcusdt_{_cal_peroid}m.csv',index_col=0)
zz.name = f"btcusdt_{_cal_peroid}m"
import datetime
#date_threshold = datetime.datetime(2020, 2, 1)
#filtered_df = z[z.index > '2020-01-01']
fct_df = zz
fct_df.index = pd.to_datetime(fct_df.index, unit='ms', utc=True)
fct_df = preprocess_data(fct_df).fillna(0)


# from volatility_factor import calc_vol_mean_reversion_factor
# from momentum_vol_factor import adaptive_momentum_factor
# bolling_band_factor = bolling_band_factor_generator(filtered_df)
# volatility_factor = calc_vol_mean_reversion_factor(filtered_df['close'])
# adaptive_momentum_factor = adaptive_momentum_factor(filtered_df)



# %%
# 3. 计算行情收益率
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
print("--- 收益率统计 ---")
print(ret.describe())

# %%
#### 4. 选择并处理单因子
# 从已生成的因子中选择一个进行回测
# 您可以取消注释其他行来测试不同的单因子
#alphas = Alphas(df=filtered_df)
final_frame = add_factor(
    fct_df, 
    factor_logic_func= calculate_ma_trend_based

)
#print(final_frame.columns)
# single_factor = volatility_factor
# single_factor = adaptive_momentum_factor

final_factor = final_frame[final_frame.name] 
#final_factor = alphas.alpha004()  # 选择 Alpha#101 作为单因子
print("\n--- 原始单因子统计 ---")
print(final_factor.describe())

#final_factor.to_csv('factor_test_data/crypto/fct001_factor.csv')



# %%
# 6. 可视化处理后的因子分布
print("\n--- 可视化最终因子分布 ---")
plt.hist(final_factor.dropna(), bins=50, alpha=0.7, label=final_factor.name)
plt.title(f"Distribution of Final Factor: {final_factor.name}")
plt.xlabel("Factor Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
#plt.show()


# %%
# 7. 因子转化为仓位
# 乘以-1是因为通常因子值越大（如突破上轨），我们预期未来价格会下跌（均值回归），所以要做空


# %%
# (可选) 保存最终因子和收益率数据，用于外部验证
# final_factor.to_csv('factor_test_data/crypto/final_factor.csv')
# ret.to_csv('factor_test_data/crypto/ret.csv')


# %%
# 8. 计算策略净值（考虑手续费）
net_values = cal_net_values(final_factor,ret)
# 9. 计算策略净值（不考虑手续费）
net_values_before_rebate = cal_net_values_before_rebate(final_factor,ret)
plt.figure(figsize=(12, 6))



normalized_close = filtered_df["close"] / filtered_df["close"].iloc[0]
plt.plot(filtered_df.index, normalized_close, label="normalized_close")
# 画净值曲线（考虑手续费）
plt.plot(net_values.index, net_values.values, label="Net Value (with fee) single", linewidth=2)

# 画净值曲线（不考虑手续费）
plt.plot(net_values_before_rebate.index, net_values_before_rebate.values, label="Net Value (before fee) single interest", linewidth=2, linestyle="--")

# 图形设置
plt.title(f"Net Value Curve Comparison - {z.name} - Factor: {final_factor.name}")
plt.xlabel("Date")
plt.ylabel("Net Value")
plt.legend()
plt.grid(True)
plt.tight_layout()




# %%
# # 10. 可视化因子预测效果 (IC分析散点图)
# plt.figure(figsize=(12, 6))
# plt.scatter(final_factor, ret, alpha=0.3)
# plt.xlabel(f'Final Factor Value ({final_factor.name})')
# plt.ylabel('Actual Returns')
# plt.title('Factor vs Actual Returns Scatter Plot')
# plt.grid(True)
# plt.show()


# %%
# 11. 计算年化夏普比率
cleaned_net_values = net_values.dropna()
sharp = calculate_sharpe_ratio_corrected(cleaned_net_values,period_minutes=_period_minutes,trading_hours=_trading_hours)

print(f"\n--- 策略表现评估 ({final_factor.name}) ---")
print(f"年化夏普比率 (Annualized Sharpe Ratio): {sharp:.4f}")

max_drawdown = calculate_max_drawdown(cleaned_net_values)


print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")

turnover = cal_turnover_annual(final_factor)

#print(f"年化换手率 (Annualized Turnover): {turnover:.2%}")

calmar = calculate_calmar_ratio(net_values)

print(f"calmar {calmar:.2f}")

# 图上方显示夏普率
plt.figtext(0.5, 0.95, f"Annualized Sharpe Ratio: {sharp:.4f}", ha="center", fontsize=12, color="blue")
plt.figtext(0.5, 0.92, f"Calmar Ratio: {calmar:.2f}", ha="center", fontsize=12, color="green")

# 图下方显示最大回撤
plt.figtext(0.5, 0.01, f"Max Drawdown: {max_drawdown:.2%}", ha="center", fontsize=12, color="red")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 预留上下空间避免覆盖
plt.show()



