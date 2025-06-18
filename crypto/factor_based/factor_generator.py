from typing import Optional
import pandas as pd
import numpy as np
from util.norm import normalize_factor

# --- 1. 定义各种因子的计算逻辑 (作为独立的函数) ---

#trend
def calculate_ma(series: pd.Series, window: int = 20) -> pd.Series:
    """计算简单移动平均线"""
    fct = normalize_factor(series["close"].rolling(window=window).mean())

    position = np.tanh(fct) * 1.5 # 平滑压缩为 [-1.5, 1.5]
    return position.where(position.abs() >1 , 0)  # 将0值替换为NaN 0.8


def _calculate_adx(series: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算ADX (Average Directional Index) - 修正版
    
    参数:
    series (pd.DataFrame): 必须包含 'high', 'low', 'close' 列的DataFrame。
    window (int): 计算ADX的周期，通常为14。

    返回:
    pd.Series: ADX指标序列。
    """
    if not all(col in series.columns for col in ['high', 'low', 'close']):
        raise ValueError("输入的数据帧 'series' 必须包含 'high', 'low', 'close' 列。")

    df = series.copy()

    df['high_prev'] = df['high'].shift(1)
    df['low_prev'] = df['low'].shift(1)
    df['close_prev'] = df['close'].shift(1)

    df['up_move'] = df['high'] - df['high_prev']
    df['down_move'] = df['low_prev'] - df['low']

    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close_prev'])
    df['tr3'] = abs(df['low'] - df['close_prev'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    smooth_plus_dm = df['+dm'].ewm(span=window, adjust=False).mean()
    smooth_minus_dm = df['-dm'].ewm(span=window, adjust=False).mean()
    smooth_tr = df['tr'].ewm(span=window, adjust=False).mean()

    plus_di = (smooth_plus_dm / smooth_tr) * 100
    minus_di = (smooth_minus_dm / smooth_tr) * 100

    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    
    dx = np.where(di_sum == 0, 0, (di_diff / di_sum) * 100)

    # --- 这里是关键的修正 ---
    # 将np.where()返回的NumPy数组dx，转换回一个带有正确索引的Pandas Series
    dx_series = pd.Series(dx, index=df.index)
    
    # 现在对新生成的Pandas Series进行计算，就不会报错了
    adx = dx_series.ewm(span=window, adjust=False).mean()
    # --- 修正结束 ---
    
    return adx.reindex(series.index)


#trend
def calculate_advanced_ma(
    series: pd.DataFrame,        # Changed to DataFrame to allow access to HLC for ATR/ADX
    fast_window: int = 20,       # Slower fast MA to reduce noise (was 10)
    slow_window: int = 60,       # Slower slow MA for more reliable trend signal (was 30)
    adx_threshold: int = 30,     # Stricter ADX to confirm stronger trends (was 25)
    atr_window: int = 14,        # Standard window for ATR calculation
    volatility_multiplier: float = 2.5 # Threshold to define "chaotic" volatility
):
    """
    Adjusted dual moving average strategy to be more robust in volatile/choppy markets.

    Key Adjustments:
    1.  Wider EMA Windows (20/60): Reduces sensitivity to noise and avoids whipsaws.
    2.  Stricter ADX Threshold (30): Ensures trades are only taken in strongly trending markets.
    3.  Volatility Filter (ATR): A new condition prevents trading when volatility is excessively
        high (chaotic), as these periods often lead to sharp reversals.
    4.  Reduced Leverage: Position size is capped at 1.0x (from 1.5x) to control risk.
    """
    
    # 1. Calculate the core signal (dual EMA difference)
    fast_ma = series["close"].ewm(span=fast_window, adjust=False).mean()
    slow_ma = series["close"].ewm(span=slow_window, adjust=False).mean()
    raw_signal = fast_ma - slow_ma
    
    # Normalize the signal to a consistent scale
    fct = normalize_factor(raw_signal)

    # 2. Define market state filters
    
    # Filter 1: Trend Filter (ADX)
    # Checks if the market is in a clear directional trend.
    adx_value = _calculate_adx(series, window=14)
    is_trending = adx_value > adx_threshold

    # Filter 2: Volatility Filter (ATR)
    # Avoids trading in chaotic periods by checking if current volatility is
    # significantly higher than its recent average.
    atr = _calculate_atr(series, window=atr_window)
    atr_long_term_avg = atr.rolling(window=slow_window * 2, min_periods=slow_window).mean()
    is_not_chaotic = atr < (atr_long_term_avg * volatility_multiplier)

    # 3. Combine filters and determine final position
    # A position is only taken if the market is trending AND not excessively volatile.
    trade_condition = is_trending & is_not_chaotic
    
    # Position sizing is now more conservative, capped at 1x leverage.
    position = np.where(trade_condition, np.tanh(fct), 0)
    
    return pd.Series(position, index=series.index)

def _calculate_atr(series: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    ATR is a measure of volatility. It requires High, Low, and Close prices.

    Args:
        series (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        window (int): The lookback period for the moving average. Typically 14.

    Returns:
        pd.Series: A series containing the ATR values.
    """
    if not all(col in series.columns for col in ['high', 'low', 'close']):
        raise ValueError("Input DataFrame 'series' must contain 'high', 'low', and 'close' columns.")

    # Get the previous day's close
    prev_close = series['close'].shift(1)

    # Calculate the three components of True Range (TR)
    # 1. Current High - Current Low
    tr1 = series['high'] - series['low']
    
    # 2. Absolute value of Current High - Previous Close
    tr2 = abs(series['high'] - prev_close)
    
    # 3. Absolute value of Current Low - Previous Close
    tr3 = abs(series['low'] - prev_close)

    # The True Range is the maximum of the three components
    # We create a temporary DataFrame to easily find the max across the rows (axis=1)
    true_range_components = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = true_range_components.max(axis=1, skipna=False)

    # Calculate the Average True Range (ATR) using an Exponential Moving Average (EMA)
    # adjust=False is used to match the common calculation method used in trading platforms.
    atr = true_range.ewm(span=window, adjust=False, min_periods=window).mean()
    
    return atr

# Example of how you would use this with the main factor function:
# Assuming 'data' is a DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
# from some_file import _calculate_adx, normalize_factor
# 
# atr_values = _calculate_atr(data, window=14)
# print(atr_values.tail())

def calculate_optimized_position(
    df: pd.DataFrame, 
    fast_window: int = 12, 
    slow_window: int = 26, 
    vol_window: int = 60,
    max_leverage: float = 1.5
) -> pd.Series:
    """
    Calculates a volatility-adjusted position based on a dual-EMA momentum signal.

    This strategy aims to improve upon a simple MA strategy by:
    1.  Using EMAs to reduce lag and improve responsiveness.
    2.  Using a fast/slow crossover system to capture momentum, reducing whipsaws.
    3.  Adjusting position size based on market volatility for better risk management.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'close' column.
        fast_window (int): Lookback window for the fast EMA.
        slow_window (int): Lookback window for the slow EMA.
        vol_window (int): Lookback window for volatility calculation.
        max_leverage (float): The maximum position size (e.g., 1.5 = 1.5x leverage).

    Returns:
        pd.Series: The calculated target position for each time step.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    # 1. Calculate Fast and Slow Exponential Moving Averages (EMAs)
    fast_ema = df["close"].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_window, adjust=False).mean()

    # 2. Create the raw momentum signal (the spread between the EMAs)
    # We normalize this spread by the slow EMA to make it comparable across different price levels.
    raw_signal = (fast_ema - slow_ema) / slow_ema

    # 3. Calculate market volatility
    # We use the standard deviation of daily log returns as our volatility measure.
    log_returns = np.log(df["close"] / df["close"].shift(1))
    market_vol = log_returns.rolling(window=vol_window).std()
    
    # To avoid extreme jumps in our volatility measure, we can use an exponentially weighted
    # standard deviation, which is smoother.
    # market_vol = log_returns.ewm(span=vol_window, adjust=False).std()

    # 4. Create the volatility-adjusted signal (the "alpha")
    # We divide our raw signal by the market volatility.
    # This increases position size in low-vol periods and decreases it in high-vol periods.
    # We replace potential infinities and NaNs that can arise from zero volatility.
    alpha_signal = raw_signal / market_vol
    alpha_signal.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha_signal.fillna(0, inplace=True) # Fill any remaining NaNs with a neutral signal

    # 5. Scale and smooth the final position
    # We use np.tanh to squash the signal into a [-1, 1] range, preventing extreme positions.
    # This adds robustness and controls risk. Then we scale by our desired max leverage.
    final_position = np.tanh(alpha_signal) * max_leverage
    
    return final_position.round(4) # Round for cleanliness

def combine_factors(factor_df: pd.DataFrame,
                    weights: dict = None,
                    method: str = "clip",
                    clip_threshold: float = 3.0) -> pd.Series:
    """
    输入多个因子，输出组合因子（路径组合）

    参数:
        factor_df: DataFrame，列是多个因子
        weights: dict[str, float]，因子组合权重（默认为等权）
        method: str, 非线性方法，支持 'tanh' 或 'clip'
        clip_threshold: float, 非线性变换的阈值

    返回:
        包含原始因子和组合因子的 DataFrame
    """
    factor_df = factor_df.copy()

    # 默认等权
    if weights is None:
        weights = {col: 1.0 for col in factor_df.columns}

    # 标准化每个因子
    for col in factor_df.columns:
        fct = factor_df[col]
        z = (fct - fct.mean()) / (fct.std() + 1e-9)
        factor_df[col] = z

    # 权重组合
    combo = sum(weights[col] * factor_df[col] for col in weights)

    # 非线性变换
    if method == "tanh":
        combo = np.tanh(combo) * clip_threshold
    elif method == "clip":
        combo = combo.clip(lower=-clip_threshold, upper=clip_threshold)

    # 只保留高置信度信号
    combo = combo.where(combo.abs() > (clip_threshold * 0.7), 0)

    return combo

#reverse
def calculate_reversal_factor_with_trend_filter(series: pd.DataFrame, 
                                                short_window: int = 60, 
                                                long_window: int = 2400,
                                                clip_value: float = 3.0) -> pd.Series:
    """
    构造带趋势过滤的改进版反转因子值。
    这个函数只生成因子，不决定仓位映射。

    参数:
        series: 包含 'close' 的 DataFrame
        short_window: 计算短期偏离的窗口大小
        long_window: 判断长期趋势的窗口大小
        clip_value: 因子值的最大绝对值

    返回:
        经过趋势过滤的反转因子值序列
    """
    close = series["close"]

    # 1. 计算原始反转因子 (使用更标准的布林带Z-score)
    short_ma = close.rolling(short_window).mean()
    short_std = close.rolling(short_window).std(ddof=1)
    zscore = (close - short_ma) / (short_std + 1e-9)
    
    # 反转逻辑：价格高于均线 -> zscore为正 -> 因子为负 (预期做空)
    raw_factor = -zscore

    # 2. 计算长期趋势过滤器
    long_ma = close.rolling(long_window).mean()
    
    # 3. 应用趋势过滤器来修正因子
    # 复制原始因子，以免修改原始数据
    filtered_factor = raw_factor.copy()

    # 当处于上升趋势时 (close > long_ma)，所有做空因子信号 (factor < 0) 都被置为0
    filtered_factor.loc[(close > long_ma) & (filtered_factor < 0)] = 0

    # 当处于下降趋势时 (close < long_ma)，所有做多因子信号 (factor > 0) 都被置为0
    filtered_factor.loc[(close < long_ma) & (filtered_factor > 0)] = 0
    
    # 4. 对最终的因子值进行裁剪，防止极端仓位
    final_factor = filtered_factor.where(filtered_factor.abs() > 2,0).clip(-clip_value, clip_value)

    return final_factor

def factor_volume_weighted_reversion(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算成交量加权的反转因子 (RSI-V)。
    
    这个因子通过结合价格变化和成交量，寻找量价背离的反转机会。
    它比标准RSI更能识别出抛售压力或购买动力的真实耗尽点。
    
    Args:
        df (pd.DataFrame): 必须包含 'close' 和 'volume' 列。
        period (int): 计算周期，通常为14。

    Returns:
        pd.Series: 成交量加权的反转因子序列，值域在0到100之间。
    """
    close = df['close']
    volume = df['volume']
    
    # 1. 计算价格变化
    delta = close.diff(1)
    
    # 2. 将价格变化分为上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 3. 核心：用成交量来加权上涨和下跌的幅度
    weighted_gain = (gain * volume).rolling(window=period).mean()
    weighted_loss = (loss * volume).rolling(window=period).mean()
    
    # 4. 计算相对强度 (RS)
    rs = weighted_gain / (weighted_loss + 1e-9)
    
    # 5. 计算最终的RSI-V值
    rsi_v = 100 - (100 / (1 + rs))
    
    norm_rsi_v = normalize_factor(rsi_v)
    return norm_rsi_v


def calculate_vwap_rank_zscore(series: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    计算 zscore(-1 * rank(close - vwap)) 因子

    参数:
        series: 包含 ['open', 'close', 'high', 'low', 'volume'] 的 DataFrame
        window: 滚动窗口大小，默认20

    返回:
        因子时间序列（已标准化）
    """

    # 计算 VWAP
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    vwap = (typical_price * series['volume']).cumsum() / series['volume'].cumsum()

    # 计算 close - vwap 差值
    diff = series['close'] - vwap

    # 时间序列 rolling rank
    rank = diff.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)

    # 取负并 zscore 标准化
    neg_rank = -rank
    zscore = (neg_rank - neg_rank.rolling(window).mean()) / (neg_rank.rolling(window).std(ddof=1) + 1e-9)
    fct = normalize_factor(zscore).where(zscore >2, 0)
    return fct
def factor_vol_adj_momentum(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    计算波动率调节的动量因子
    
    Rationale:
    - 捕捉趋势，同时在市场动荡时降低风险敞口。
    - 目标是平滑收益曲线，减少剧烈回撤，从而提升Sharpe Ratio。
    """
    df_factor = df.copy()
    
    # 计算N日动量（累计对数收益率）
    momentum = df_factor['log_return'].rolling(window=n).sum()
    
    # 使用已经计算好的波动率（或重新计算一个不同周期的）
    # 为避免除以零，给波动率加上一个极小值
    vol = df_factor['volatility'] + 1e-9 
    
    factor = momentum / vol
    
    return normalize_factor(factor)

def calculate_momentum(series: pd.Series, period: int = 10, window: int = 100, threshold: float = 2.7) -> pd.Series:
    """
    计算标准化后的动量因子，并去除低置信度噪声

    参数:
        series: 价格序列
        period: 动量周期
        window: 标准化的 rolling 窗口大小
        threshold: 置信度截断值（越高保留越强信号）

    返回:
        信号序列，只保留绝对值大于阈值的信号，其余为 0
    """

    series = series["close"]
    momentum = series.diff(period) / series.shift(period)

    # 标准化
    zscore = (momentum - momentum.rolling(window).mean()) / (momentum.rolling(window).std(ddof=1) + 1e-9)

    # 映射 [-3, 3] 再截断
    normed = zscore.clip(-3, 3)

    # 只保留高置信度信号
    filtered = normed.where(normed < -2.7, 0)

    return filtered.fillna(0)



def factor_bollinger_power(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """
    计算布林带力量反转因子
    
    Rationale:
    - 捕捉短期过度反应后的均值回归机会。
    - 当价格极端偏离均值时（例如，恐慌性抛售），往往是反向操作的好时机。
    - 这类策略通常胜率高，但需要严格的风控。
    """
    df_factor = df.copy()
    
    sma = df_factor['close'].rolling(window=n).mean()
    stddev = df_factor['close'].rolling(window=n).std()
    
    # 标准化价格与中轨的距离
    fct = normalize_factor((df_factor['close'] - sma) / (k * stddev + 1e-9))
    
    position = fct.where(fct.abs() > 2.8, 0)  # 只保留高置信度信号
    return position

# --- 2. 创建一个通用的因子添加函数 ---

def add_factor(
    df: pd.DataFrame, 
    factor_logic_func, 
    factor_name: Optional[str] = None, 
    base_col: str = 'close', 
    **kwargs
) -> pd.DataFrame:
    """
    一个通用的函数，用于计算并添加因子到DataFrame。

    参数:
    - df (pd.DataFrame): 原始 OHLCV DataFrame。
    - factor_logic_func (function): 用于计算因子的函数 (例如 calculate_ma)。
    - factor_name (str): 新因子列的名称。
    - base_col (str): 用于计算因子的基础列，默认为'close'。
    - **kwargs: 传递给因子计算函数的其他参数 (例如 window=5)。

    返回:
    - pd.DataFrame: 添加了新因子并对齐好列的DataFrame。
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入 DataFrame 必须包含以下列: {required_columns}")
    if base_col not in df.columns:
        raise ValueError(f"用于计算的基础列 '{base_col}' 不在DataFrame中。")

    df_copy = df.copy()

    factor_name = factor_logic_func.__name__ if factor_name is None else factor_name
    # 使用传入的函数和参数进行计算
    df_copy[factor_name] = factor_logic_func(df_copy, **kwargs)

    # 对齐输出列
    final_df = df_copy.fillna(0)

    final_df.name = factor_name
    return final_df


import pandas as pd
import numpy as np

# 假设已存在以下辅助函数
# from your_utils import _calculate_adx, _calculate_atr, normalize_factor

def calculate_multi_period_momentum_filter_hourly(
    series: pd.DataFrame,
    short_window: int = 72,       # 短期动能均线 (约3天)
    medium_window: int = 168,     # 中期动能均线 (约7天)
    long_window: int = 720 * 6,       # 长期趋势过滤均线 (关键, 约180天)
    adx_window: int = 20,         # ADX周期 (略微加长以平滑小时线噪音)
    adx_threshold: int = 28,      # 更严格的ADX阈值，过滤掉弱趋势
    atr_window: int = 20,         # ATR波动率计算周期
    volatility_threshold: float = 2.5 # 波动率过滤阈值
) -> pd.Series:
    """
    一个专门为【小时线】周期优化的多周期过滤动量因子。

    策略逻辑:
    1.  长期趋势过滤 (Regime Filter): 使用约30日均线 (720根小时K线) 判断整体市场方向。
    2.  市场状态过滤 (State Filter): 使用ADX和ATR判断当前是否为“值得交易”的趋势行情。
    3.  入场信号 (Entry Signal): 当以上所有条件满足时，使用约3日和7日的EMA交叉作为交易方向。
    """

    # --- 1. 计算所有需要的指标 ---
    
    # 长期趋势过滤器 (SMA更稳定)
    long_term_ma = series['close'].rolling(window=long_window).mean()

    # 中短期动能信号 (EMA更灵敏)
    short_term_ma = series['close'].ewm(span=short_window, adjust=False).mean()
    medium_term_ma = series['close'].ewm(span=medium_window, adjust=False).mean()
    raw_momentum_signal = short_term_ma - medium_term_ma

    # 市场状态过滤器指标
    adx = _calculate_adx(series, window=adx_window)
    atr = _calculate_atr(series, window=atr_window)
    atr_ma = atr.rolling(window=long_window, min_periods=medium_window).mean()

    # --- 2. 定义过滤条件 (布尔值) ---

    # 主要趋势方向
    is_long_term_bull = series['close'] > long_term_ma
    is_long_term_bear = series['close'] < long_term_ma

    # 市场状态
    is_trending = adx > adx_threshold
    is_not_chaotic = atr < (atr_ma * volatility_threshold)

    # 动能方向
    has_bullish_momentum = raw_momentum_signal > 0
    has_bearish_momentum = raw_momentum_signal < 0

    # --- 3. 组合逻辑，生成最终仓位 ---

    # 做多条件: 长期牛市 + 市场有趋势 + 波动不极端 + 中短期看涨动能
    can_go_long = is_long_term_bull & is_trending & is_not_chaotic & has_bullish_momentum

    # 做空条件: 长期熊市 + 市场有趋势 + 波动不极端 + 中短期看跌动能
    can_go_short = is_long_term_bear & is_trending & is_not_chaotic & has_bearish_momentum

    # 初始化仓位为0
    position = pd.Series(0.0, index=series.index)
    
    # 使用归一化后的信号来决定仓位大小，使得信号更稳定
    scaled_signal = normalize_factor(raw_momentum_signal)
    
    position.loc[can_go_long] = np.tanh(scaled_signal[can_go_long]) # 使用tanh平滑仓位大小
    position.loc[can_go_short] = np.tanh(scaled_signal[can_go_short])

    # 对最终仓位进行轻微平滑，防止信号过于频繁地在0和非0之间跳动
    final_position = position.ewm(span=5, adjust=False).mean()

    return final_position


def generate_signals_with_threshold(factor_series: pd.Series, long_threshold: float, short_threshold: float, is_reversal: bool = False) -> pd.Series:
    """
    根据绝对阈值生成信号，引入中性区。
    
    Rationale:
    - 避免在信号不强不弱时进行无效交易，减少交易成本和噪音。
    - 只在有高把握的机会时才入场。
    """
    signal = pd.Series(0, index=factor_series.index)
    
    if not is_reversal: # 动量因子
        signal[factor_series > long_threshold] = 1
        signal[factor_series < short_threshold] = -1
    else: # 反转因子
        signal[factor_series < long_threshold] = 1  # 因子值极低时做多
        signal[factor_series > short_threshold] = -1 # 因子值极高时做空
        
    return signal

# --- 示例用法 ---

# 使用与之前相同的模拟数据 stock_df

# a. 使用通用框架添加 MA 因子
# ma_factor_df = add_factor(
#     stock_df, 
#     factor_logic_func=calculate_ma, 
#     factor_name='MA_factor', 
#     window=5
# )
# print("--- 使用通用框架添加 MA 因子 ---")
# print(ma_factor_df.head(10))

# print("\n" + "="*40 + "\n")

# # b. 使用通用框架添加 Momentum 因子
# momentum_factor_df = add_factor(
#     stock_df,
#     factor_logic_func=calculate_momentum,
#     factor_name='Momentum_10d',
#     base_col='close', # 明确指定使用收盘价
#     period=10
# )
# print("--- 使用通用框架添加 Momentum 因子 ---")
# print(momentum_factor_df.head(15))

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """计算后续因子所需的基础数据"""
    # 使用对数收益率，更适合金融时间序列
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 计算波动率（例如，过去20天的日收益率标准差）
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    # 计算平均成交量
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    
    df.dropna(inplace=True)
    return df




# class Alphas:
#     """
#     实现 WorldQuant 101 Alphas，并提供可选的因子标准化功能。

#     本类接收一个标准的 OHLCV 金融时间序列数据作为输入，并提供
#     方法来计算 101 个 Alpha 因子。

#     参数:
#     df (pd.DataFrame): 
#         输入的 DataFrame，必须包含以下列:
#         'open', 'high', 'low', 'close', 'volume'
#         索引必须是时间序列索引 (例如 pd.DatetimeIndex)。
    
#     normalize_alphas (bool, optional): 
#         如果为 True，所有生成的因子将进行时间序列标准化（滚动Z-score）。
#         默认为 False。
        
#     norm_window (int, optional):
#         进行时间序列标准化时使用的滚动窗口大小。
#         默认为 126 (约半年)。

#     使用方法:
#     >>> df = create_dummy_data()
#     >>> # 不进行标准化
#     >>> alpha_gen_raw = Alphas(df, normalize_alphas=False)
#     >>> raw_alpha1 = alpha_gen_raw.alpha001()
#     >>> # 进行标准化
#     >>> alpha_gen_norm = Alphas(df, normalize_alphas=True)
#     >>> normalized_alpha1 = alpha_gen_norm.alpha001()
#     """
#     def __init__(self, df: pd.DataFrame, normalize_alphas: bool = True, norm_window: int = 126):
#         # 1. 数据校验和预处理
#         required_cols = ['open', 'high', 'low', 'close', 'volume']
#         for col in required_cols:
#             if col not in df.columns:
#                 raise ValueError(f"输入的 DataFrame 缺少必需列: '{col}'")
        
#         self.df = df.copy()
        
#         # 2. 方便地访问数据列
#         self.open = self.df['open']
#         self.high = self.df['high']
#         self.low = self.df['low']
#         self.close = self.df['close']
#         self.volume = self.df['volume']
        
#         # 3. 预计算常用数据
#         self.returns = self.close.pct_change()
#         self.vwap = (self.close * self.volume).cumsum() / self.volume.cumsum()
        
#         # 4. 标准化设置
#         self.normalize_alphas = normalize_alphas
#         self.norm_window = norm_window

#     # ---------------------------------------------------------------- #
#     # 辅助函数 (Helper Functions)
#     # ---------------------------------------------------------------- #

#     def _normalize_factor(self, series: pd.Series) -> pd.Series:
#         """
#         对因子进行时间序列标准化 (滚动Z-score)。
#         """
#         # 使用 expanding() 来处理窗口初期的数据不足问题
    
#         return normalize_factor(series, window=self.norm_window)

#     def _finalize_alpha(self, raw_alpha: pd.Series) -> pd.Series:
#         """
#         在返回最终 Alpha 前，根据设置决定是否进行标准化。
#         """
#         if self.normalize_alphas:
#             return self._normalize_factor(raw_alpha)
#         return raw_alpha

#     # ... 其他辅助函数 _delay, _rank, _correlation 等保持不变 ...
#     def _delay(self, series: pd.Series, period: int) -> pd.Series:
#         return series.shift(period)
#     def _delta(self, series: pd.Series, period: int) -> pd.Series:
#         return series.diff(period)
#     def _rank(self, series: pd.Series) -> pd.Series:
#         return series.rank(pct=True)
#     def _correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
#         return series1.rolling(window).corr(series2)
#     def _stddev(self, series: pd.Series, window: int) -> pd.Series:
#         return series.rolling(window).std()
#     def _ts_max(self, series: pd.Series, window: int) -> pd.Series:
#         return series.rolling(window).max()
#     def _ts_argmax(self, series: pd.Series, window: int) -> pd.Series:
#         return series.rolling(window).apply(np.argmax, raw=True)
#     def _signed_power(self, series: pd.Series, power: float) -> pd.Series:
#         return np.sign(series) * (np.abs(series) ** power)
#     def ts_rank(self, series: pd.Series, window: int) -> pd.Series:
#         def rank_last(window_data):
#             return pd.Series(window_data).rank(pct=True).iloc[-1]
#         return series.rolling(window).apply(rank_last, raw=False)


#     # ---------------------------------------------------------------- #
#     # Alpha 因子实现 (示例)
#     # ---------------------------------------------------------------- #

#     def alpha001(self) -> pd.Series:
#         """
#         Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
#         """
#         inner = np.where(self.returns < 0, self._stddev(self.returns, 20), self.close)
#         signed_power = self._signed_power(pd.Series(inner, index=self.df.index), 2.)
#         ts_argmax = self._ts_argmax(signed_power, 5)
#         raw_alpha = self._rank(ts_argmax) - 0.5
        
#         return self._finalize_alpha(raw_alpha)

#     def alpha002(self) -> pd.Series:
#         """
#         Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
#         """
#         log_volume = np.log1p(self.volume) 
#         part1 = self._rank(self._delta(log_volume, 2))
#         part2 = self._rank((self.close - self.open) / self.open)
#         corr = self._correlation(part1, part2, 6)
#         raw_alpha = -1 * corr

#         return self._finalize_alpha(raw_alpha)

#     def alpha003(self) -> pd.Series:
#         """
#         Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
#         """
#         part1 = self._rank(self.open)
#         part2 = self._rank(self.volume)
#         corr = self._correlation(part1, part2, 10)
#         raw_alpha = -1 * corr
        
#         return self._finalize_alpha(raw_alpha)

#     def alpha004(self) -> pd.Series:
#         """
#         Alpha#4: (-1 * Ts_Rank(rank(low), 9))
#         """
#         ranked_low = self._rank(self.low)
#         raw_alpha = self.ts_rank(ranked_low, 9) * -1
        
#         return self._finalize_alpha(raw_alpha)
        
#     def alpha101(self) -> pd.Series:
#         """
#         Alpha#101: ((close - open) / ((high - low) + .001))
#         """
#         raw_alpha = (self.close - self.open) / ((self.high - self.low) + 0.001)
        
#         return self._finalize_alpha(raw_alpha)