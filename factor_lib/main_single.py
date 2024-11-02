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
#1. 读取行情数据
z = pd.read_csv(f'data/crypto/btcusdt_{_period_minutes}m.csv',index_col=0)
z.name = f"btcusdt_{_period_minutes}m"
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
# factors = [
#     bolling_band_factor,
#     volatility_factor,
#     adaptive_momentum_factor
# ]

# %%
# 6. 计算行情收益率
ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
print(ret.describe())

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




