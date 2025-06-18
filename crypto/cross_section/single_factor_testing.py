# %%
# 在文件最开头添加以下代码
import os
import sys
import glob # 用于文件路径匹配

# 正确的写法：
# 请确保这个路径是正确的项目根目录
project_root = "/Users/wanting/Downloads/multifactor_quant_learning"
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.norm import normalize_factor
from util.sharpe_calculatio import cal_sharp_random,calculate_sharpe_ratio_corrected
from util import calculate_max_drawdown
from factor_based.calculate_net_vaules import cal_turnover_annual # 引用年化换手率计算
from factor_based.calculate_net_vaules import *
pd.plotting.register_matplotlib_converters()
from factor_generator import *


# %%
# =============================================================================
# 0. 策略参数设置
# =============================================================================
_period_minutes = 60
_trading_hours = 24
DATA_DIR = 'data/crypto' # 数据目录
FEE_RATE = 0.0005 # 双边手续费率 (0.05%)
NUM_QUANTILES = 5 # 分层数量，5代表五分位法

# %%
# =============================================================================
# 1. 数据加载与预处理 (修改为多资产)
# =============================================================================

def load_and_process_all_assets(data_dir, period_minutes):
    """
    加载并处理指定目录下所有资产的行情数据。
    """
    path_pattern = os.path.join(data_dir, f'*_{period_minutes}h.csv')
    all_files = glob.glob(path_pattern)
    
    if not all_files:
        raise FileNotFoundError(f"在目录 '{data_dir}' 中没有找到匹配 '_{period_minutes}h.csv' 的文件")

    all_assets_df = []
    print(f"找到 {len(all_files)} 个资产文件，正在加载...")

    for file_path in all_files:
        try:
            # 从文件名中提取 symbol
            symbol = os.path.basename(file_path).replace(f'_{period_minutes}h.csv', '').upper()
            
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index, unit='ms', utc=True)
            df = preprocess_data(df) # 应用您原来的预处理函数
            df['symbol'] = symbol
            all_assets_df.append(df)
            print(f"  - 已加载并处理 {symbol}")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 合并所有数据到一个大的 DataFrame 中
    panel_df = pd.concat(all_assets_df, axis=0)
    panel_df = panel_df.set_index(['symbol'], append=True).swaplevel(0, 1) # 创建 (symbol, timestamp) 多级索引
    panel_df = panel_df.sort_index()
    
    # 筛选共同的时间范围，确保截面数据完整
    common_dates = panel_df.index.get_level_values(1).value_counts()
    min_assets_count = len(all_files) * 0.8 # 要求至少80%的资产在该时间点有数据
    valid_dates = common_dates[common_dates >= min_assets_count].index
    panel_df = panel_df[panel_df.index.get_level_values(1).isin(valid_dates)]

    print(f"\n数据加载完成。总数据条数: {len(panel_df)}, 唯一资产数: {panel_df.index.get_level_values(0).nunique()}")
    return panel_df

# 加载所有资产数据
panel_data = load_and_process_all_assets(DATA_DIR, 1)
panel_data = panel_data.loc[panel_data.index.get_level_values(1) > pd.Timestamp("2020-06-01").tz_localize("UTC")]


# %%
# =============================================================================
# 2. 因子计算 (应用于所有资产)
# =============================================================================
print("\n--- 正在为所有资产计算因子... ---")

# 使用 groupby().apply() 为每个资产计算因子
# 注意：add_factor 需要能处理传入的 Series/DataFrame
# 您的 add_factor 看起来是直接操作 DataFrame，所以这样用是合适的
factor_series = panel_data.groupby(level='symbol').apply(
    lambda x: add_factor(x.reset_index(level='symbol', drop=True), factor_logic_func=calculate_optimized_position_v2)
)

# 提取因子列并构建因子面板
factor_name = factor_series.name
factor_panel = factor_series[factor_name].unstack(level='symbol')
print(f"因子 '{factor_name}' 计算完成。")
print("\n--- 原始因子面板预览 ---")
print(factor_panel.tail())

# %%
# =============================================================================
# 3. 计算所有资产的收益率
# =============================================================================
print("\n--- 正在计算所有资产的收益率... ---")
# 计算每个资产的未来一期收益率
returns_panel = panel_data['close'].unstack(level='symbol').pct_change().shift(-1).fillna(0)
print("收益率面板计算完成。")


# %%
# =============================================================================
# 4. 截面策略核心：构建多空组合仓位
# =============================================================================
def build_cross_sectional_positions(factor_panel, num_quantiles=5):
    """
    根据因子面板构建截面多空仓位。
    - 做多因子值最高的组
    - 做空因子值最低的组
    """
    # 1. 横向排名（在每个时间点上对所有资产的因子值进行排名）
    ranks = factor_panel.rank(axis=1, method='first', na_option='keep')
    
    # 2. 确定做多和做空的资产
    # 资产总数（非空）
    asset_counts = ranks.count(axis=1)
    # 做多阈值：排名 > 总数 * (n-1)/n
    long_threshold = asset_counts * (num_quantiles - 1) / num_quantiles
    # 做空阈值：排名 <= 总数 / n
    short_threshold = asset_counts / num_quantiles
    
    # 3. 生成仓位信号（1 for long, -1 for short, 0 for neutral）
    longs = (ranks.gt(long_threshold, axis=0)).astype(int)
    shorts = (ranks.le(short_threshold, axis=0)).astype(int) * -1
    positions = longs + shorts
    
    # 4. 仓位归一化（市场中性 & 资金中性）
    # 计算每期做多和做空的资产数量
    num_longs = (positions > 0).sum(axis=1)
    num_shorts = (positions < 0).sum(axis=1)
    
    # 防止除以零
    num_longs[num_longs == 0] = 1
    num_shorts[num_shorts == 0] = 1
    
    # 多头仓位 = 1 / 多头数量, 空头仓位 = -1 / 空头数量
    positions[positions > 0] = positions[positions > 0].div(num_longs, axis=0)
    positions[positions < 0] = positions[positions < 0].div(num_shorts, axis=0)
    
    # 总体仓位调整为总杠杆为1（多头0.5，空头0.5）
    positions = positions / 2
    
    return positions.fillna(0)

print(f"\n--- 正在构建截面仓位 (分 {NUM_QUANTILES} 组)... ---")
final_positions = build_cross_sectional_positions(factor_panel, num_quantiles=NUM_QUANTILES)
print("仓位构建完成。")
print("\n--- 最终仓位表示例 (前5行) ---")
print(final_positions.head())


# %%
# =============================================================================
# 5. 可视化因子分布
# =============================================================================
print("\n--- 可视化所有资产的因子合并分布 ---")
plt.figure(figsize=(10, 5))
# 将所有因子值合并到一个Series中并去除NaN
all_factor_values = factor_panel.stack().dropna()
plt.hist(all_factor_values, bins=100, alpha=0.7, label=factor_name)
plt.title(f"Distribution of Pooled Factor: {factor_name}")
plt.xlabel("Factor Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


# %%
# =============================================================================
# 6. 回测与绩效计算
# =============================================================================

# 确保所有 DataFrame 的索引对齐
common_index = final_positions.index.intersection(returns_panel.index)
final_positions = final_positions.loc[common_index]
returns_panel = returns_panel.loc[common_index]

# 计算投资组合的每日收益（不考虑费用）
# 仓位在 t-1 时刻决定，作用于 t-1 到 t 的收益
# 所以用 t-1 的仓位乘以 t 时刻的收益
portfolio_returns_before_fee = (final_positions.shift(1) * returns_panel).sum(axis=1)

# 计算换手率和手续费
# 换手率 = sum(|pos_t - pos_{t-1}|)
turnover = (final_positions - final_positions.shift(1).fillna(0)).abs().sum(axis=1)
transaction_costs = turnover * FEE_RATE

# 计算扣除手续费后的净收益
portfolio_returns = portfolio_returns_before_fee - transaction_costs

# 计算净值曲线
net_values = (1 + portfolio_returns).cumprod()
net_values_before_rebate = (1 + portfolio_returns_before_fee).cumprod()

# 计算等权基准的净值
benchmark_returns = returns_panel.mean(axis=1)
benchmark_net_value = (1 + benchmark_returns).cumprod()


# %%
# =============================================================================
# 7. 策略表现可视化
# =============================================================================
plt.figure(figsize=(14, 7))

# 画净值曲线
plt.plot(net_values.index, net_values, label="Net Value (with fee)", linewidth=2)
plt.plot(net_values_before_rebate.index, net_values_before_rebate, label="Net Value (before fee)", linewidth=1.5, linestyle="--")
plt.plot(benchmark_net_value.index, benchmark_net_value, label="Equal-Weighted Benchmark", linewidth=1.5, linestyle=":", color='gray')


# 图形设置
plt.title(f"Cross-Sectional Strategy Performance - Factor: {factor_name}")
plt.xlabel("Date")
plt.ylabel("Cumulative Net Value")
plt.legend()
plt.grid(True)
plt.yscale('log') # 使用对数坐标轴，更好地观察长期趋势


# %%
# =============================================================================
# 8. 计算和展示绩效指标
# =============================================================================
cleaned_net_values = net_values.dropna()
sharp = calculate_sharpe_ratio_corrected(cleaned_net_values, period_minutes=_period_minutes, trading_hours=_trading_hours)

print(f"\n--- 截面策略表现评估 ({factor_name}) ---")
print(f"年化夏普比率 (Annualized Sharpe Ratio): {sharp:.4f}")

max_drawdown = calculate_max_drawdown(cleaned_net_values)
print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")

# 计算年化换手率
# 每日换手率的年化 = 日均换手率 * 年交易期数
periods_per_day = (_trading_hours * 60) / _period_minutes
periods_per_year = periods_per_day * 365
annualized_turnover = turnover.mean() * periods_per_year
print(f"年化换手率 (Annualized Turnover): {annualized_turnover:.2%}")

# 在图上显示关键指标
plt.figtext(0.5, 0.95, f"Annualized Sharpe Ratio: {sharp:.4f}", ha="center", fontsize=12, color="blue")
plt.figtext(0.5, 0.01, f"Max Drawdown: {max_drawdown:.2%} | Annualized Turnover: {annualized_turnover:.2%}", ha="center", fontsize=12, color="red")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()