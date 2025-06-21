
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
from util.sharpe_calculatio import *
from calculate_net_vaules import cal_net_values, cal_net_values_before_rebate,cal_net_values_compounded
from combine_factor import *
# from verify_risk_orthogonalization import risk_orthogonalization # 不再需要风险正交
pd.plotting.register_matplotlib_converters()
from factor_generator import *
import logging
logger = logging.getLogger("sample")
logger.addHandler(logging.FileHandler("./crypto/logging.txt"))
logger.setLevel(logging.INFO)

# %%
# 0 data preprocess
_period_minutes = 60
_trading_hours = 24
_coin = "eth"
# %%
#1. 读取行情数据
z = pd.read_csv(f'data/crypto/{_coin}usdt_{_period_minutes}m.csv',index_col=0)
z.name = f"{_coin}usdt_{_period_minutes}m"
import datetime
#date_threshold = datetime.datetime(2020, 2, 1)
#filtered_df = z[z.index > '2020-01-01']
filtered_df = z
filtered_df.index = pd.to_datetime(filtered_df.index, unit='ms', utc=True)
filtered_df = preprocess_data(filtered_df)
#filtered_df = filtered_df.loc[filtered_df.index > pd.Timestamp("2024-06-01").tz_localize("UTC")]
#z.head()

# %%
# 2. 生成各种备选因子


# from volatility_factor import calc_vol_mean_reversion_factor
# from momentum_vol_factor import adaptive_momentum_factor
# bolling_band_factor = bolling_band_factor_generator(filtered_df)
# volatility_factor = calc_vol_mean_reversion_factor(filtered_df['close'])
# adaptive_momentum_factor = adaptive_momentum_factor(filtered_df)



# %%
# 3. 计算行情收益率

ret = filtered_df['close'].pct_change().shift(-1).fillna(0)
#print("--- 收益率统计 ---")
#print(ret.describe())

# %%
#### 4. 选择并处理单因子
# 从已生成的因子中选择一个进行回测
# 您可以取消注释其他行来测试不同的单因子
#alphas = Alphas(df=filtered_df)
final_frame = add_factor(
    filtered_df, 
    factor_logic_func=calculate_optimized_position_v2 , 
)

final_frame = add_factor(
    final_frame, 
    factor_logic_func=calculate_multi_period_momentum_filter_hourly , 
)
final_frame = add_factor(
    final_frame, 
    factor_logic_func=greed_factor , 
)
final_frame = add_factor(
    final_frame,    
    factor_logic_func=fct001 ,
)
final_frame = add_factor(
    final_frame,
    factor_logic_func=calculate_ma,
)
final_frame = add_factor(
    final_frame,
    factor_logic_func=calculate_momentum,
)
final_frame = add_factor(
    final_frame, 
    factor_logic_func=laziness_factor #Share 1.3
)
final_frame = add_factor(
    final_frame, 
    factor_logic_func=fear_factor
)

final_frame = add_factor(
    final_frame, 
    factor_logic_func=fct007
)

final_frame = add_factor(
    final_frame, 
    factor_logic_func=mean_revert_when_neutral_and_stable
)
final_frame = add_factor(
    final_frame,    
    factor_logic_func=fct003 ,
)

final_frame = add_factor(
    final_frame,    
    factor_logic_func=fct004 ,
)

final_frame = add_factor(
    final_frame, 
    factor_logic_func=create_trend_following_vol_factor
)

final_frame = add_factor(
    final_frame, 
    factor_logic_func=factor_bollinger_power
)

# single_factor = volatility_factor
# single_factor = adaptive_momentum_factor

#print("\n--- 多因子组合 ---",final_frame.columns[-6:])
final_factor = combine_factors_lightgbm(final_frame, factor_cols=["mean_revert_when_neutral_and_stable","create_trend_following_vol_factor","factor_bollinger_power","calculate_ma","fct001","calculate_optimized_position_v2","greed_factor","calculate_multi_period_momentum_filter_hourly","laziness_factor"],weights=[ 0.359000,0.516833,0.124167])
# final_factor = combine_factors_linear(final_frame, factor_cols=final_frame.columns[-6:],weights=[0.2,0.2,0.2,0.2,0.2,0.2]) 
#final_factor = alphas.alpha004()  # 选择 Alpha#101 作为单因子
#print("\n--- 多因子统计 ---")
#print(final_factor.describe())

#final_factor.to_csv(f'factor_test_data/crypto/final_factor{_coin}.csv')





# %%
# # 6. 可视化处理后的因子分布
# print("\n--- 可视化最终因子分布 ---")
# plt.hist(final_factor.dropna(), bins=50, alpha=0.7, label=final_factor.name)
# plt.title(f"Distribution of Final Factor: {final_factor.name}")
# plt.xlabel("Factor Value")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)
# plt.show()


# %%
# 7. 因子转化为仓位
# 乘以-1是因为通常因子值越大（如突破上轨），我们预期未来价格会下跌（均值回归），所以要做空


# %%
# (可选) 保存最终因子和收益率数据，用于外部验证
# final_factor.to_csv('factor_test_data/crypto/final_factor.csv')
# ret.to_csv('factor_test_data/crypto/ret.csv')


# %%
# 8. 计算策略净值（考虑手续费）¥¥¥¥¥¥¥4
net_values = cal_net_values_compounded(final_factor,ret)
#print(net_values.values)  # numpy array
# 9. 计算策略净值（不考虑手续费）
net_values_before_rebate = cal_net_values(final_factor,ret)
plt.figure(figsize=(12, 6))


normalized_close = filtered_df["close"] / filtered_df["close"].iloc[0]  # 归一化收盘价


# normalized_close = normalized_close.loc[normalized_close.index > pd.Timestamp("2024-10-01").tz_localize("UTC")]
# net_values = net_values.loc[net_values.index > pd.Timestamp("2024-10-01").tz_localize("UTC")]
# net_values_before_rebate = net_values_before_rebate.loc[net_values_before_rebate.index > pd.Timestamp("2024-10-01").tz_localize("UTC")]

plt.plot(normalized_close.index, normalized_close, label="normalized_close")
# 画净值曲线（考虑手续费）
plt.plot(net_values.index, net_values.values, label="Net Value (with fee) compounded", linewidth=2)
#print(net_values)

# 画净值曲线（不考虑手续费）
plt.plot(net_values_before_rebate.index, net_values_before_rebate.values, label="Net Value (with fee) single interest", linewidth=2)
# 图形设置
plt.title(f"Net Value Curve Comparison - {z.name} - Factor: {final_factor.name}")
plt.xlabel("Date")
plt.ylabel("Net Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# fct001_ortho                             0.516833
# calculate_ma_ortho                       0.359000
# calculate_optimized_position_v2_ortho    0.124167


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
logger.info(f"年化夏普比率 (Annualized Sharpe Ratio): {sharp:.4f}")


sharp_single = calculate_sharpe_ratio_corrected(net_values_before_rebate,period_minutes=_period_minutes,trading_hours=_trading_hours)
logger.info(f"年化夏普比率 (Annualized Sharpe Ratio single interest): {sharp_single:.4f}")

# 12. 计算max drawdown
from util.max_drawdown import calculate_max_drawdown
max_drawdown = calculate_max_drawdown(cleaned_net_values)
logger.info(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")

calmar_ratio = calculate_calmar_ratio(net_values)
logger.info(f"Calmar Ratio: {calmar_ratio:.2f}")

plt.figtext(0.5, 0.95, f"Annualized Sharpe Ratio(compounded): {sharp:.4f}", ha="center", fontsize=12, color="blue")


plt.figtext(0.5, 0.93, f"Annualized Sharpe Ratio(single): {sharp_single:.4f}", ha="center", fontsize=12, color="blue")

# 图下方显示最大回撤
plt.figtext(0.5, 0.01, f"Max Drawdown: {max_drawdown:.2%}", ha="center", fontsize=12, color="red")

plt.figtext(0.5, 0.98, f"Calmar Ratio: {calmar_ratio:.4f}", ha="center", fontsize=12, color="green")

ar = calculate_annualized_return(net_values)
plt.figtext(0.5, 0.05, f"annual return: {ar:.2%}", ha="center", fontsize=12, color="blue")
logger.info(f"annual return: {ar:.2%}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 预留上下空间避免覆盖
plt.show()