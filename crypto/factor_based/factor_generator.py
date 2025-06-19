from typing import Optional
import pandas as pd
import numpy as np
from util.norm import *

# --- 1. 定义各种因子的计算逻辑 (作为独立的函数) ---

#1.1trend vol AS0.77
#你可以用n个周期后的ret作为label，还可以再雕琢一下，用n个周期后不高于/不低于某个价格位置作为label
# filter 震荡市
def calculate_ma(series: pd.Series, window: int = 20*24) -> pd.Series:
    """计算简单移动平均线"""
    fct = normalize_factor(series["close"].rolling(window=window).mean().ewm(span=5*24).mean())

    position = np.tanh(fct) * 1.5 # 平滑压缩为 [-1.5, 1.5]
    return position  # 将0值替换为NaN 0.8

def fct001(df: pd.DataFrame) -> pd.Series:
    """One-liner version of the factor calculation."""
    #ratio = ratio.clip(lower=0.8, upper=1.2)  # 控制在合理区间内，防止极端值
    
    raw_factor = -df['volatility']**2 * np.log(df['volatility']**2 * np.log(np.log(df['close'] / (df['high'] + 1e-9))**2) * np.log(df['close'] / (df['high'] + 1e-9)))
    #factor = pd.Series(winsorize(raw_factor, limits=[0.01, 0.01]), index=df.index)
    #
    factor = normalize_factor(raw_factor)  # Normalize the factor to handle NaNs and scale it
    # threshold = factor.abs().quantile(0.999)
    # filtered = factor.where(factor.abs() <= threshold, -factor)

    #neg_factor = normalize_factor(-raw_factor)
    return factor # Smoothing with EWM

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


#1.2trend
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

#1.3momentum
def calculate_optimized_position(
    df: pd.DataFrame, 
    fast_window: int = 7, 
    slow_window: int = 2*7, 
    vol_window: int = 30,
    max_leverage: float = 1.5
) -> pd.Series:
    """
    Calculates a volatility-adjusted position based on a dual-EMA momentum signal.
    
    V2 Optimizations:
    1.  Signal Inversion: Inverted the final signal to fix the negative correlation issue observed in backtests.
    2.  Return Value Bug Fix: Correctly returns the scaled and smoothed final position.
    3.  Smoother Volatility: Uses EWM for volatility calculation to reduce jerky position changes.
    4.  Signal Stability: Adds an epsilon to the volatility denominator to prevent division by zero.
    5.  Removed dependency on an external `normalize_factor` function, making the logic self-contained.

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

    # 1. Calculate Fast and Slow EMAs (No change here)
    fast_ema = df["close"].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_window, adjust=False).mean()

    # 2. Create the raw momentum signal (No change here)
    raw_signal = (fast_ema - slow_ema) / slow_ema

    # 3. [OPTIMIZED] Calculate a smoother market volatility
    log_returns = np.log(df["close"] / df["close"].shift(1))
    # Using ewm().std() for a smoother, more responsive volatility measure
    market_vol = log_returns.ewm(span=vol_window, adjust=False).std()
    
    # 4. [OPTIMIZED] Create a more stable volatility-adjusted signal
    # Add a small epsilon to the denominator to prevent division by near-zero values
    epsilon = 1e-8
    alpha_signal = raw_signal / (market_vol + epsilon)
    
    # Fill any NaNs that might arise at the beginning
    alpha_signal.fillna(0, inplace=True)

    # 5. [OPTIMIZED & BUG FIX] Scale, smooth, and invert the final position
    # np.tanh squashes the signal into a [-1, 1] range, controlling risk from extreme alpha values.
    # We then scale by our desired max leverage.
    scaled_position = np.tanh(alpha_signal) * max_leverage
    
    # [CRITICAL FIX] Invert the signal. Since the original trend-following logic
    # resulted in performance opposite to the market trend, we invert the position.
    # A positive signal (uptrend) will now correctly result in a positive (long) position.
    # If your backtest system inverts signals, you might need to remove this negative sign.
    # But based on your chart, this inversion is necessary.
    final_position = scaled_position

    # We return the correctly calculated, scaled, and inverted position.
    return final_position

from scipy.stats.mstats import winsorize

def _ts_rank(series: pd.Series, window: int = 10) -> pd.Series:
    """
    时间序列排名。计算每个点在过去`window`天内的百分比排名。
    
    :param series: 输入时间序列。
    :param window: 滚动窗口大小。
    :return: 排名后的时间序列 (0到1之间)。
    """
    return series.rolling(window=window).rank(pct=True)

def _ts_corr(series1: pd.Series, series2: pd.Series, window: int = 10) -> pd.Series:
    """
    时间序列相关性。计算两个序列在过去`window`天内的滚动相关性。
    
    :param series1: 第一个输入时间序列。
    :param series2: 第二个输入时间序列。
    :param window: 滚动窗口大小。
    :return: 滚动相关性序列。
    """
    return series1.rolling(window=window).corr(series2)
#0.6 反应过度，衰竭延续 0.9
#########yuming


def fct005(df: pd.DataFrame, window: int = 10, delay: int = 1) -> pd.Series:
    """
    排名 1 的 Alpha 因子

    因子表达式:
    add(_ts_rank(close), _ts_corr(sqrt(ts_delay(_ts_rank(mul(_ts_corr(high, high), sub(volatility, taker buy quote asset volume))))), _ts_rank(open)))
    
    Args:
        df (pd.DataFrame): 
            输入数据框，必须包含 'close', 'high', 'volatility', 
            'taker buy quote asset volume', 'open' 列。
        window (int, optional): 时间序列操作的回看窗口。默认为 10。
        delay (int, optional): ts_delay 的延迟期。默认为 1。

    Returns:
        pd.Series: 计算出的因子值序列。
    """
    required_cols = ['close', 'high', 'volatility', 'taker buy quote asset volume', 'open']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"输入 DataFrame 必须包含以下列: {required_cols}")

    # 定义 _ts_rank 的 lambda 函数
    _ts_rank = lambda s: s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # _ts_corr(high, high) 在窗口期内方差不为0时恒为1
    ts_corr_high_high = df['high'].rolling(window).corr(df['high']).fillna(1.0)
    sub_vol_taker = df['volatility'] - df['taker buy quote asset volume']
    mul_term = ts_corr_high_high * sub_vol_taker
    
    ts_rank_mul = _ts_rank(mul_term)
    ts_delay_rank = ts_rank_mul.shift(delay)
    
    # 为避免对负数开方，使用 clip(lower=0)
    sqrt_delayed_rank = np.sqrt(ts_delay_rank.clip(lower=0))

    ts_rank_open = _ts_rank(df['open'])
    
    correlation_term = sqrt_delayed_rank.rolling(window).corr(ts_rank_open)
    ts_rank_close = _ts_rank(df['close'])
    
    alpha_factor = ts_rank_close + correlation_term
    return normalize_factor(-alpha_factor)


def fct006(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    排名 2 的 Alpha 因子

    因子表达式:
    add(_ts_rank(close), _ts_corr(_ts_rank(close), _ts_rank(open)))
    
    Args:
        df (pd.DataFrame): 输入数据框，必须包含 'open' 和 'close' 列。
        window (int, optional): 时间序列操作的回看窗口。默认为 10。

    Returns:
        pd.Series: 计算出的因子值序列。
    """
    if 'open' not in df.columns or 'close' not in df.columns:
        raise ValueError("输入 DataFrame 必须包含 'open' 和 'close' 列。")

    ts_rank_close = df['close'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    ts_rank_open = df['open'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    ts_correlation = ts_rank_close.rolling(window).corr(ts_rank_open)

    alpha_factor = ts_rank_close + ts_correlation
    return normalize_factor(-alpha_factor)


def fct007(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    排名 3 的 Alpha 因子

    因子表达式:
    add(_ts_rank(close), _ts_corr(_ts_corr(sqrt(sqrt(log(taker buy base asset volume))), _ts_rank(open)), _ts_rank(open)))
    
    Args:
        df (pd.DataFrame): 
            输入数据框，必须包含 'close', 'taker buy base asset volume', 'open' 列。
        window (int, optional): 时间序列操作的回看窗口。默认为 10。

    Returns:
        pd.Series: 计算出的因子值序列。
    """
    required_cols = ['close', 'taker buy base asset volume', 'open']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"输入 DataFrame 必须包含以下列: {required_cols}")

    _ts_rank = lambda s: s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # 处理 log(0) 或 log(负数)，并处理 log(x) < 0 (当0<x<1) 的情况
    log_taker_buy = np.log(df['taker buy base asset volume'].clip(lower=1e-9))
    sqrt1_log = np.sqrt(log_taker_buy.clip(lower=0))
    sqrt2_log = np.sqrt(sqrt1_log)

    ts_rank_open = _ts_rank(df['open'])
    
    inner_corr = sqrt2_log.rolling(window).corr(ts_rank_open)
    outer_corr = inner_corr.rolling(window).corr(ts_rank_open)
    
    ts_rank_close = _ts_rank(df['close'])
    
    alpha_factor = ts_rank_close + outer_corr
    return normalize_factor(alpha_factor)


def fct008(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    排名 4 的 Alpha 因子

    因子表达式:
    add(_ts_rank(open), _ts_corr(div(_ts_rank(_ts_rank(quote asset volume)), neg(log(-0.814))), _ts_rank(open)))
    
    注意: 表达式中 neg(log(-0.814)) 在实数域无定义。此处假设其意图为
    neg(log(abs(-0.814)))，并将其作为常数处理。

    Args:
        df (pd.DataFrame): 输入数据框，必须包含 'open' 和 'quote asset volume' 列。
        window (int, optional): 时间序列操作的回看窗口。默认为 10。

    Returns:
        pd.Series: 计算出的因子值序列。
    """
    required_cols = ['open', 'quote asset volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"输入 DataFrame 必须包含以下列: {required_cols}")

    _ts_rank = lambda s: s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # 计算常数部分
    constant_val = -np.log(abs(-0.814))  #约 0.2058

    ts_rank_qav = _ts_rank(df['quote asset volume'])
    ts_rank_ts_rank_qav = _ts_rank(ts_rank_qav) # 嵌套 _ts_rank

    div_term = ts_rank_ts_rank_qav / constant_val

    ts_rank_open = _ts_rank(df['open'])
    
    correlation_term = div_term.rolling(window).corr(ts_rank_open)
    
    alpha_factor = ts_rank_open + correlation_term
    return alpha_factor


def fct009(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    排名 5 的 Alpha 因子

    因子表达式:
    add(_ts_rank(close), _ts_corr(taker buy quote asset volume, _ts_rank(low)))
    
    Args:
        df (pd.DataFrame): 
            输入数据框，必须包含 'close', 'taker buy quote asset volume', 'low' 列。
        window (int, optional): 时间序列操作的回看窗口。默认为 10。

    Returns:
        pd.Series: 计算出的因子值序列。
    """
    required_cols = ['close', 'taker buy quote asset volume', 'low']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"输入 DataFrame 必须包含以下列: {required_cols}")
    
    ts_rank_low = df['low'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    ts_rank_close = df['close'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # 注意：这里直接使用原始的 'taker buy quote asset volume'
    correlation_term = df['taker buy quote asset volume'].rolling(window).corr(ts_rank_low)
    
    alpha_factor = ts_rank_close + correlation_term
    return normalize_factor(alpha_factor)


def calculate_fct001_with_trend_filter(df: pd.DataFrame) -> pd.Series:
    """
    计算fct001，并使用50日均线作为趋势过滤器，
    以避免在连续恐慌下跌行情中产生错误信号。
    """
    # 1. 定义趋势过滤器
    # 计算50日简单移动平均线
    sma_50 = df['close'].rolling(window=7*24, min_periods=20).mean() # min_periods防止早期太多NaN
    
    # 创建一个布尔掩码：当收盘价 > 50日线时为True，表示处于上升趋势
    is_in_uptrend = df['close'] > sma_50

    # 2. 计算原始因子
    # (为了代码整洁，我们把之前的逻辑放在这里)
    volatility = df['volatility']
    close = df['close']
    high = df['high']
    
    # 防止除以0或log(0)的错误
    safe_high = high + 1e-9
    log_c_h = np.log(close / safe_high)

    # 复杂的内部项，用try-except处理可能的数学错误（如log负数）
    try:
        inner_log = np.log(volatility**2 * np.log(log_c_h**2))
        raw_factor = -volatility**2 * inner_log * log_c_h
    except (ValueError, FloatingPointError):
        # 如果计算出错，则生成一个全为NaN的Series
        raw_factor = pd.Series(np.nan, index=df.index)

    # 3. 应用过滤器
    # 将所有不处于上升趋势的日子的因子值设为0（或np.nan）
    # 这样做，因子就不会在下降趋势中给出任何信号
    filtered_factor = raw_factor.copy()
    filtered_factor[~is_in_uptrend] = 0 # ~is_in_uptrend 表示选择处于下降趋势的日子

    filtered_factor.name = 'fct001_filtered'
    return normalize_factor(filtered_factor)

def fct002(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the modified complex factor in a single line.

    Formula: volatility**2 * log( log(close / high**2) * log(close / high) )

    :param df: DataFrame with 'close', 'high', and 'volatility' columns.
    :return: A pandas Series with the calculated factor.
    """
    # Using a small epsilon to avoid issues if high is ever zero
    epsilon = 1e-9
    
    factor = -df['volatility']**2 * np.log(np.log(df['close'] / (df['high']**2 + epsilon)) * np.log(df['close'] / (df['high'] + epsilon)))
    
    return normalize_factor(factor)
#reverse


def fct003(
    df: pd.DataFrame,
    interpretation: int = 1,
    return_period_short: int = 1,
    return_period_long: int = 10,
    delay_period: int = 1
) -> pd.Series:
    """
    Calculates a complex factor based on an interpretation of the ambiguous formula:
    'mul(sub(returns, returns), ts_delay(log_return))'

    :param df: DataFrame with 'close', 'high', 'low' columns.
    :param interpretation: 
        1 for 'Momentum Change' interpretation.
        2 for 'Dual-Period Momentum' interpretation.
    :param return_period_short: Lookback period for short-term returns.
    :param return_period_long: Lookback period for long-term returns (for interpretation 2).
    :param delay_period: The delay period for the log return signal.
    :return: A pandas Series with the calculated and normalized factor.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    # Calculate the common components first
    log_return = np.log(df['close'] / df['close'].shift(1))
    delayed_log_return = log_return.shift(delay_period)

    # --- Apply selected interpretation ---
    
    if interpretation == 1:
        # Interpretation 1: Momentum Change (returns.diff() * delayed_log_return)
        # Measures the "acceleration" or "deceleration" of returns.
        returns = df['close'].pct_change(return_period_short)
        momentum_change = returns.diff(1)
        factor = momentum_change * delayed_log_return

    elif interpretation == 2:
        # Interpretation 2: Dual-Period Momentum Crossover
        # Compares short-term momentum with long-term momentum.
        short_term_returns = df['close'].pct_change(return_period_short)
        long_term_returns = df['close'].pct_change(return_period_long)
        momentum_spread = short_term_returns - long_term_returns
        factor = momentum_spread * delayed_log_return
        
    else:
        raise ValueError("Invalid interpretation. Choose 1 or 2.")

    # --- Final Step: Normalize the factor ---
    # The `normalize_factor` function handles NaNs and scales the factor.
    return normalize_factor(-factor)


def fct004(df: pd.DataFrame, window: int = 20*24) -> pd.Series:
    """
    Computes a factor by multiplying historical volatility by lagged returns.

    The formula is an interpretation of `mul(ts_std(log(0.109)), ts_delay(returns))`.
    A literal interpretation would yield zero, as the standard deviation of a
    constant (`log(0.109)`) is zero. This implementation assumes `0.109` was a
    placeholder for the 'close' price series to create a meaningful financial
    factor, a common practice in quantitative finance.

    The implemented formula is:
    `mul(ts_std(log(close), window), ts_delay(returns, 1))`

    Args:
        df (pd.DataFrame):
            Input DataFrame containing financial data. Must include a 'close' column.
        window (int, optional):
            The look-back period for the rolling standard deviation. Defaults to 20.

    Returns:
        pd.Series:
            A Series containing the calculated factor values. The initial values
            will be NaN due to the rolling window and lagging.
            
    Raises:
        ValueError: If the input DataFrame does not contain a 'close' column.
    """
    if 'close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    # Calculate returns from the 'close' price series
    # returns = (price_t / price_{t-1}) - 1
    returns = df['close'].pct_change()

    # ts_delay(returns): Lag the returns series by one period
    delayed_returns = returns.shift(1)

    # log(close): Calculate the natural logarithm of the close price
    # We assume 'close' prices are positive.
    log_close = np.log(df['close'])

    # ts_std(log(close), window): Calculate the rolling standard deviation
    # of the log-price series. This is a common measure of volatility.
    log_volatility = log_close.rolling(window=window).std()

    # mul(...): Multiply the volatility by the lagged returns
    factor = -log_volatility * delayed_returns
    
    factor.name = 'fct004'

    factor = normalize_factor(factor)
    return factor


# --- 因子函数生成 ---

def fct010(df: pd.DataFrame) -> pd.Series:
    """
    排名 1 的因子实现。
    
    原始公式: sub(log(_ts_rank(returns)), abs(_ts_corr(avg_volume_7, -0.578)))
    假设窗口期: 10
    
    :param df: 包含 'close', 'volume' 列的DataFrame。
    :return: 计算并标准化后的因子Series。
    """
    WINDOW = 14*24  # 假设的时间序列操作窗口
    
    returns = df['close'].pct_change()
    avg_volume_7 = df['avg_volume_7']
    # 创建一个与df索引对齐的常数Series
    const_series = pd.Series(-0.578, index=df.index)
    
    term1 = np.log(_ts_rank(returns, window=WINDOW))
    # 注意: 与常数的相关性将返回NaN
    term2 = np.abs(_ts_corr(avg_volume_7, const_series, window=WINDOW))
    
    factor = term1 - term2
    
    return normalize_factor(factor)


def fct011(df: pd.DataFrame) -> pd.Series:
    """
    排名 2 的因子实现。
    
    原始公式: mul(abs(log_return), log(_ts_corr(0.880, open)))
    假设窗口期: 10
    
    :param df: 包含 'open', 'close' 列的DataFrame。
    :return: 计算并标准化后的因子Series。
    """
    WINDOW = 10
    
    log_return = np.log(df['close'] / df['close'].shift(1))
    const_series = pd.Series(0.880, index=df.index)

    term1 = np.abs(log_return)
    # 注意: 与常数的相关性将返回NaN，取对数后仍然是NaN
    term2 = np.log(_ts_corr(df['open'], const_series, window=WINDOW))
    
    factor = term1 * term2
    
    return normalize_factor(factor)


def fct012(df: pd.DataFrame) -> pd.Series:
    """
    排名 3 的因子实现。
    
    原始公式: mul(_ts_corr(_ts_rank(sqrt(low)), _ts_rank(neg(open))), abs(log_return))
    假设窗口期: 10
    
    :param df: 包含 'low', 'open', 'close' 列的DataFrame。
    :return: 计算并标准化后的因子Series。
    """
    WINDOW = 10*24
    
    log_return = np.abs(np.log(df['close'] / df['close'].shift(1)))
    
    ts_rank_low = _ts_rank(np.sqrt(df['low']), window=WINDOW)
    ts_rank_open_neg = _ts_rank(-df['open'], window=WINDOW)
    
    corr_term = _ts_corr(ts_rank_low, ts_rank_open_neg, window=WINDOW)
    
    factor = normalize_factor(-corr_term * log_return)
    
    return factor.where(factor.abs() >0.5,0)


def fct013(df: pd.DataFrame) -> pd.Series:
    """
    排名 4 的因子实现。
    
    原始公式: div(avg_volume_20, avg_volume_20)
    
    :param df: 包含 'volume' 列的DataFrame。
    :return: 计算并标准化后的因子Series。
    """
    avg_volume_20 = df['volume'].rolling(window=20*7).mean()
    
    # 这个因子在avg_volume_20非零的地方等于1，否则为NaN。
    # 这是一个几乎没有信息的因子。
    factor = avg_volume_20 / avg_volume_20
    
    # 结果将是全为0的Series，因为标准化后常数序列的标准差为0。
    return normalize_factor(factor)


def fct014(df: pd.DataFrame) -> pd.Series:
    """
    排名 5 的因子实现。
    
    原始公式: mul(abs(log_return), _ts_corr(_ts_rank(close), _ts_corr(0.326, log_return)))
    假设窗口期: 10
    
    :param df: 包含 'close' 列的DataFrame。
    :return: 计算并标准化后的因子Series。
    """
    WINDOW = 10*24
    
    log_return = np.log(df['close'] / df['close'].shift(1))
    const_series = pd.Series(0.326, index=df.index)
    
    term1 = np.abs(log_return)
    
    # 内部相关性: 与常数相关，返回NaN
    inner_corr = _ts_corr(log_return, const_series, window=WINDOW)
    
    ts_rank_close = _ts_rank(df['close'], window=WINDOW)
    
    # 外部相关性: 与一个NaN序列相关，同样返回NaN
    outer_corr = _ts_corr(ts_rank_close, inner_corr, window=WINDOW)
    
    factor = term1 * outer_corr
    
    return normalize_factor(factor)

    
    return normalize_factor(factor.rolling(window=window).mean())
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


def calculate_optimized_position_v2(
    df: pd.DataFrame, 
    fast_window: int = 24, #sensitivity to short-term trends
    slow_window: int = 7*24, 
    vol_window: int = 90*24,
    max_leverage: float = 1.5,
    strategy: str = 'risk_averse' # Options: 'risk_averse', 'trend_acceleration', 'regime_filter'
) -> pd.Series:
    """
    Calculates a target position based on a dual-EMA momentum signal, with
    selectable volatility-based sizing strategies.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'close' column.
        fast_window (int): Lookback window for the fast EMA.
        slow_window (int): Lookback window for the slow EMA.
        vol_window (int): Lookback window for volatility calculation.
        max_leverage (float): The maximum position size (e.g., 1.5 = 1.5x leverage).
        strategy (str): The position sizing strategy to use.
            - 'risk_averse': Original logic. Position is inversely proportional to volatility.
            - 'trend_acceleration': Position is directly proportional to volatility.
            - 'regime_filter': Uses volatility level to decide whether to take a signal.

    Returns:
        pd.Series: The calculated target position for each time step.
    """
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    # 1. Calculate Fast and Slow Exponential Moving Averages (EMAs)
    fast_ema = df["close"].ewm(span=fast_window, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_window, adjust=False).mean()

    # 2. Create the raw momentum signal (normalized spread)
    raw_signal = (fast_ema - slow_ema) / slow_ema

    # 3. Calculate market volatility (smoother EWM of standard deviation)
    log_returns = np.log(df["close"] / df["close"].shift(1))
    market_vol = log_returns.ewm(span=vol_window, adjust=False).std()

    # 4. Apply the chosen sizing strategy to create the "alpha"
    alpha_signal = pd.Series(np.nan, index=df.index) # Initialize

    if strategy == 'risk_averse':
        # Original logic: Decrease position in high vol
        alpha_signal = raw_signal / (market_vol + 1e-10)  # Add epsilon to avoid division by zero
        
    elif strategy == 'trend_acceleration':
        # New logic: INCREASE position in high vol to capture trends
        # We scale by a constant (e.g., 100) to bring the signal into a reasonable
        # range before the tanh squashing function. This is a tunable parameter.
        alpha_signal = raw_signal 
    
    elif strategy == 'regime_filter':
        # Hybrid logic: Use vol to filter signal
        # Only take a signal if volatility is above its own moving average (i.e., in a high-vol state)
        vol_ma = market_vol.ewm(span=vol_window * 2, adjust=False).mean()
        in_high_vol_regime = (market_vol > vol_ma)
        
        # Scale the raw signal; this becomes the base for our position
        # A simple scalar can be used, or it can be a function of vol itself
        scaled_raw_signal = raw_signal * 10 # Tunable scalar
        
        # Only apply the signal when in the high-vol regime
        alpha_signal = scaled_raw_signal.where(in_high_vol_regime, 0)
        
    else:
        raise ValueError("Invalid strategy. Choose from 'risk_averse', 'trend_acceleration', 'regime_filter'.")

    alpha_signal.replace([np.inf, -np.inf], np.nan, inplace=True)
    alpha_signal.fillna(0, inplace=True)

    # 5. Scale and smooth the final position
    # np.tanh squashes the signal into a [-1, 1] range, controlling risk.
    final_position = normalize_factor(alpha_signal) 
    #final_position = np.tanh(final_position) * 3
    return final_position  # Round for cleanliness



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


def calculate_vwap_rank_zscore(series: pd.DataFrame, window: int = 180) -> pd.Series:
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
    # neg_rank = rank
    fct = normalize_factor(rank)
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



def factor_bollinger_power(df: pd.DataFrame, n: int = 120*24, k: float = 3.0) -> pd.DataFrame:
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
    
    return fct 

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
    short_window: int = 3*24,       # 短期动能均线 (约3天)
    medium_window: int = 7*24,     # 中期动能均线 (约7天)
    long_window: int = 120*24,       # 长期趋势过滤均线 (关键, 约180天)
    adx_window: int = 20,         # ADX周期 (略微加长以平滑小时线噪音)
    adx_threshold: int = 8,      # 更严格的ADX阈值，过滤掉弱趋势
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
    position.loc[can_go_short] = np.tanh(scaled_signal[can_go_short])  # 做空时取负值

    # 对最终仓位进行轻微平滑，防止信号过于频繁地在0和非0之间跳动
    #final_position = position.where(position.abs() > 0.1, 0)  # 只保留绝对值大于0.1的信号

    return position


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


def create_volatility_band_factor(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    short_window: int = 5, 
    long_window: int = 14
) -> pd.Series:
    """
    计算波动率期限结构比率，并将其限制在[2, 3]的有效区间内。

    该因子通过计算短期波动率与长期波动率的比值，来捕捉持续的极端行情。
    它只在比值落入[2, 3]区间时输出信号，其他情况均为0。

    参数:
    df (pd.DataFrame): 包含价格数据的时间序列DataFrame。
                         **注意：该函数设计用于日线数据**，因为窗口期以天为单位。
    price_col (str): DataFrame中表示价格的列名，默认为'close'。
    short_window (int): 计算短期滚动波动率的窗口期（天数）。
    long_window (int): 计算长期滚动波动率的窗口期（天数）。

    返回:
    pd.Series: 波动率带通因子序列。值为0或在[2, 3]区间内。
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不在DataFrame中。")

    # 1. 确保数据是日线级别（如果输入是更高频率，需要先重采样）
    # 为确保计算准确，我们假设输入已经是日线数据。
    
    # 2. 计算日收益率
    daily_returns = df[price_col].pct_change()

    # 3. 计算短期和长期滚动波动率（年化标准差）
    # 乘以 sqrt(252) 是为了年化，但对于比率计算，此步骤可以省略，因为会被约掉。
    # 这里我们使用原始标准差，更直观。
    short_term_vol = daily_returns.rolling(window=short_window).std()
    long_term_vol = daily_returns.rolling(window=long_window).std()

    # 4. 计算波动率比率
    # 核心逻辑：当前波动状态 / 历史常态
    vol_ratio = short_term_vol / long_term_vol

    # 5. 应用[2, 3]带通滤波器
    # 使用 np.where 实现条件逻辑：
    # 条件：vol_ratio >= 2 且 vol_ratio <= 3
    # 如果为真，则因子值 = vol_ratio
    # 如果为假，则因子值 = 0
    condition = (vol_ratio >= 2) & (vol_ratio <= 3)
    band_factor = np.where(condition, vol_ratio, 0)

    # 6. 转换为Pandas Series并处理初始NaN值
    factor_series = pd.Series(band_factor, index=df.index).fillna(0)
    
    # 7. 命名Series以便识别
    factor_series.name = f'vol_band_{short_window}_{long_window}'
    
    return factor_series


def create_volatility_band_factor_three_windows(
    df: pd.DataFrame,
    price_col: str = 'close',
    short_window: int = 5*24,
    medium_window: int = 14*24,
    long_window: int = 60*24,
    avg_window: int = 252*24,
    base_lower: float = 2.0,
    base_upper: float = 3.0
) -> pd.Series:
    """
    基于短期与中期波动率比值，结合长期波动率平均水平动态调整阈值的波动率带通因子，归一化到[-3,3]，
    并只保留绝对值大于2的信号捕获极端行情。
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不在DataFrame中。")

    daily_returns = df[price_col].pct_change()

    short_vol = daily_returns.rolling(window=short_window).std()
    medium_vol = daily_returns.rolling(window=medium_window).std()
    long_vol = daily_returns.rolling(window=long_window).std()

    avg_long_vol = long_vol.rolling(window=avg_window, min_periods=20).mean()

    vol_ratio = short_vol / medium_vol

    scale_factor = avg_long_vol / avg_long_vol.mean()

    factor_series = pd.Series(scale_factor, index=df.index).fillna(0)

    # 归一化因子到[-3,3]
    normalized_factor = normalize_factor(factor_series)

    # 只保留绝对值大于2的极端行情信号，其余置0
    extreme_factor = normalized_factor.where(normalized_factor.abs() > 2, 0)

    extreme_factor.name = f'vol_band_norm_extreme_{short_window}_{medium_window}_{long_window}_{avg_window}'

    return extreme_factor




def create_trend_following_vol_factor(
    df: pd.DataFrame,
    price_col: str = 'close',
    short_window: int = 5*24,
    medium_window: int = 14*24,
    long_window: int = 60*24, # 保持参数以便比较，但可能不会全用到
) -> pd.Series:
    """
    一个结合了波动率强度和趋势方向的趋势跟踪因子。
    当市场活跃且存在明显趋势时，因子值会变大。
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不在DataFrame中。")

    daily_returns = df[price_col].pct_change()

    # 1. 计算波动率比率作为“市场活跃度”或“信号强度”
    short_vol = daily_returns.rolling(window=short_window).std()
    medium_vol = daily_returns.rolling(window=medium_window).std()
    # 使用 clip(lower=1e-8) 防止除以零
    vol_ratio = short_vol / medium_vol.clip(lower=1e-8)
    
    # 2. 计算中期动量作为“趋势方向”
    # 用 medium_window 周期内的价格变化百分比来定义趋势
    momentum = df[price_col].pct_change(periods=medium_window)
    
    # 3. 结合强度与方向
    # np.sign(momentum) 会得到 +1 (上涨趋势), -1 (下跌趋势), 或 0 (无变化)
    # 乘以 vol_ratio，使得在市场更活跃时，因子绝对值更大
    trend_factor = vol_ratio * np.sign(momentum)
    
    # 对因子进行平滑处理，使其信号更稳定
    trend_factor_smoothed = trend_factor.ewm(span=short_window, adjust=False).mean()
    
    trend_factor_smoothed.name = f'trend_vol_factor_{short_window}_{medium_window}'
    
    return trend_factor_smoothed.fillna(0)


def create_breakout_trend_factor(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 20*24, # 常用的布林带窗口
    n_std: float = 2.0   # 标准差倍数
) -> pd.Series:
    """
    基于布林带通道突破的经典趋势跟踪因子。
    突破上轨产生+1信号，跌破下轨产生-1信号，并持续持有直到趋势反转。
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不在DataFrame中。")
    
    # 1. 计算布林带的中、上、下轨
    sma = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    upper_band = sma + (rolling_std * n_std)
    lower_band = sma - (rolling_std * n_std)
    
    # 2. 生成原始突破信号
    signal = pd.Series(np.nan, index=df.index)
    signal[df[price_col] > upper_band] = 1  # 突破上轨，做多
    signal[df[price_col] < lower_band] = -1 # 跌破下轨，做空
    
    # 3. 定义趋势结束条件（可选，但推荐）
    # 当价格回到中轨时，我们认为趋势可能暂停或反转，清除信号
    # 做多时，如果价格跌破中轨，平仓
    long_exit = (df[price_col] < sma)
    # 做空时，如果价格涨过中轨，平仓
    short_exit = (df[price_col] > sma)
    
    # 将平仓信号位置设置为0
    signal[long_exit | short_exit] = 0
    
    # 4. 持续持有信号（这是实现“跟随”的关键）
    # 使用 forward fill 填充 NaN，使得信号在整个趋势期间保持不变
    factor = signal.ffill().fillna(0)
    
    factor.name = f'breakout_trend_factor_{window}_{n_std}'
    return normalize_factor(factor).where(factor.abs() > 2, 0)  # 只保留绝对值大于0.1的信号


def calculate_complex_factor_corrected(df: pd.DataFrame) -> pd.Series:
    """
    Calculates a corrected version of the factor to avoid math errors.

    Corrected Formula: volatility**2 * log(abs((log(close / high))**3))

    This version takes the absolute value before the final logarithm
    to ensure the input is positive.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'close', 'high', and 'volatility' columns.

    Returns:
        pd.Series: The calculated factor series.
    """
    df = df.copy()
    if 'volatility' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'volatility' column.")

    # Step 1: div(close, high)
    div_ch = df['close'] / (df['high']+1e-9)  # 防止除以零

    # Step 2: log(div(close, high))
    log_div = np.log(div_ch)

    # Step 3: (log(div(close, high)))**3
    log_div_cubed = log_div ** 3
    
    # Step 4: log of the ABSOLUTE VALUE of the result from Step 3
    # This correction prevents math errors.
    log_of_abs_cubed = np.log(np.abs(log_div_cubed))
    
    # Step 5: Multiply by volatility squared
    factor = -df['volatility'].pow(2) * log_of_abs_cubed
    
    factor.name = 'complex_factor_corrected'
    return normalize_factor(factor)

# 使用示例 (假设你有一个名为 df 的 DataFrame)
# df = pd.read_csv('your_crypto_data.csv', index_col='timestamp', parse_dates=True)
# trend_factor = create_trend_following_vol_factor(df)
# trend_factor.plot()
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
    df['volatility'] = df['log_return'].rolling(window=20*24).std()
    
    # 计算平均成交量
    df['avg_volume_20'] = df['volume'].rolling(window=20*24).std()

    df['avg_volume_7'] = df['volume'].rolling(window=7*24).std()

    df["returns"] = df["close"].pct_change().fillna(0)

    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    df["returns_7_std"] = df["returns"].rolling(window=7*24).std()
    df["returns_20_std"] = df["returns"].rolling(window=20*24).std()

    df["trades_std"] = df['number of trades'].rolling(window=20*24).std()
    
    df = df.loc[df.index > pd.Timestamp("2020-10-01").tz_localize("UTC")]
    df.dropna(inplace=True)
    return df


def calculate_volatility_zscore(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    window: int = 24
) -> pd.Series:
    """
    计算波动率Z-Score因子。

    该因子通过将当前收益率与其历史滚动波动率进行比较，
    来量化价格运动的极端程度。

    因子值 > 3 或 < -3 通常表示发生了远超近期常态的极端行情。

    参数:
    df (pd.DataFrame): 包含价格数据的时间序列DataFrame，索引必须是时间类型。
    price_col (str): DataFrame中表示价格的列名，默认为'close'。
    window (int): 计算历史滚动波动率的回看窗口期，默认为24（例如24小时）。

    返回:
    pd.Series: 与输入DataFrame具有相同索引的因子序列。
    """
    if price_col not in df.columns:
        raise ValueError(f"列 '{price_col}' 不在DataFrame中。")

    # 1. 计算收益率
    returns = df[price_col].pct_change()

    # 2. 计算历史滚动波动率（标准差）
    # 使用 .shift(1) 来确保我们用的是 t-1 到 t-N 的数据，避免前视偏差
    historical_vol = returns.rolling(window=window).std().shift(1)

    # 3. 计算因子值 (当前收益率 / 历史波动率)
    zscore_factor = returns / historical_vol

    # 4. 处理边界和异常情况
    # - 将除以零产生的无穷大值替换为0 (表示无有效信号)
    # - 将窗口初期产生的NaN值填充为0
    zscore_factor = zscore_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5. 命名Series
    zscore_factor.name = f'vol_zscore_{window}'

    return zscore_factor




def gelu(x: pd.Series) -> pd.Series:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
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
#     def _ts_rank(self, series: pd.Series, window: int) -> pd.Series:
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
#         raw_alpha = self._ts_rank(ranked_low, 9) * -1
        
#         return self._finalize_alpha(raw_alpha)
        
#     def alpha101(self) -> pd.Series:
#         """
#         Alpha#101: ((close - open) / ((high - low) + .001))
#         """
#         raw_alpha = (self.close - self.open) / ((self.high - self.low) + 0.001)
        
#         return self._finalize_alpha(raw_alpha)


