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
# 4 .计算annual夏普比率
# 输入为每日净值序列
def cal_sharp(net_values: np.array) -> float:
    '''计算夏普比率，risk free rate为无风险年化收益率，trading_days是1年的交易日'''
    risk_free_rate = 0.05  # 无风险收益率
    trading_days = 252  # 一年的交易日

    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]  # 计算相对收益率
    mean_return = np.mean(returns)  # 平均收益率

    # 计算超额收益率
    excess_return = mean_return - risk_free_rate  # 超额收益率

    # 计算收益率的标准差
    std_dev = np.std(returns,ddof = 1)  # 收益率的标准差

    # 计算夏普比率
    sharpe_ratio = (excess_return * trading_days) / std_dev  # 年化夏普比率

    return sharpe_ratio

# %%
#all day trading
def cal_annual_sharpe_15mins(net_values: np.array, risk_free_rate: float = 0.05) -> float:
    # 计算15分钟收益率
    returns = np.diff(net_values) / net_values[:-1]

    # 计算日收益率
    daily_return = (1 + returns) ** 96 - 1

    # 计算年化收益率
    annual_return = (1 + np.mean(daily_return)) ** 252 - 1

    # 计算年化标准差
    annual_std_dev = np.std(returns) * np.sqrt(252 * 96)

    # 计算超额收益
    excess_return = annual_return - risk_free_rate

    # 计算年化夏普比率
    if annual_std_dev == 0:
        return np.nan  # 避免除以零
    sharpe_ratio = excess_return / annual_std_dev

    return sharpe_ratio



def cal_sharp_random(net_values: np.array, period_minutes: int = 15, trading_hours: int = 4) -> float:
    '''计算年化夏普比率
    Args:
        net_values: np.array, 净值序列
        period_minutes: int, 数据周期（分钟）
        trading_hours: int, 每天交易小时数
    Returns:
        float: 年化夏普比率
    '''
    risk_free_rate = 0.03  # 年化无风险收益率
    
    # 计算年化系数
    periods_per_day = (trading_hours * 60) // period_minutes  # 每天的周期数
    trading_days = 252  # 一年的交易日数
    annual_factor = periods_per_day * trading_days  # 年化系数
    
    # 计算收益率
    returns = np.diff(net_values) / net_values[:-1]
    
    # 确保returns不为空
    if len(returns) == 0:
        return np.nan
    
    # 计算年化收益率和年化标准差
    mean_return = np.mean(returns) * annual_factor
    std_dev = np.std(returns, ddof=1) * np.sqrt(annual_factor)
    
    # 确保标准差不为零
    if std_dev == 0:
        return np.nan
        
    # 计算超额收益率
    excess_return = mean_return - risk_free_rate
    
    # 计算夏普比率
    sharpe_ratio = excess_return / std_dev
    
    return sharpe_ratio

# %%
# 6. 计算行情收益率
ret = filtered_df['close'].shift(-1) / filtered_df['close'] - 1
ret.describe()

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

