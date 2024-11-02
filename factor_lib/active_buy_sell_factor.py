import pandas as pd
import numpy as np
from util.norm import normalize_factor

def calculate_act_factors(data:pd.DataFrame,ret:pd.Series) -> pd.Series:
    """
    计算主动买卖因子(ACT)
    
    Parameters:
    data: pandas.DataFrame with columns:
        - date: 交易日期
        - large_mid_buy: 大单和中单主动买入金额
        - large_mid_sell: 大单和中单主动卖出金额
        - small_buy: 小单主动买入金额
        - small_sell: 小单主动卖出金额
        - returns: 当日收益率
    
    Returns:
    dict: 包含高收益日和低收益日的ACT因子平均值
    """
    #
    data['returns'] = data['close'].pct_change().fillna(0)

    # 计算正向ACT (大单和中单)
    data['act_positive'] = (data['large_mid_buy'] - data['large_mid_sell']) / \
                          (data['large_mid_buy'] + data['large_mid_sell'])
    
    # 计算负向ACT (小单)
    data['act_negative'] = (data['small_buy'] - data['small_sell']) / \
                          (data['small_buy'] + data['small_sell'])
    
    # 使用rolling window获取过去20个交易日
    def analyze_window(window):
        if len(window) < 20:
            return pd.Series({'high_return_act_positive': np.nan, 
                            'low_return_act_negative': np.nan})
        
        # 获取收益率最高和最低的日期
        high_return_dates = window.nlargest(1, 'returns').index
        low_return_dates = window.nsmallest(1, 'returns').index
        
        # 计算高收益日的正向ACT平均值
        high_return_act = window.loc[high_return_dates, 'act_positive'].mean()
        
        # 计算低收益日的负向ACT平均值
        low_return_act = window.loc[low_return_dates, 'act_negative'].mean()
        
        return pd.Series({
            'high_return_act_positive': high_return_act,
            'low_return_act_negative': low_return_act
        })
    
    # 应用滚动窗口分析
    factor = data.rolling(window=20, min_periods=20).apply(analyze_window)
    
    return normalize_factor(factor)

# def example_usage():
#     """
#     示例使用
#     """
#     # 创建示例数据
#     dates = pd.date_range(start='2024-01-01', periods=30, freq='B')
#     sample_data = pd.DataFrame({
#         'date': dates,
#         'large_mid_buy': np.random.uniform(1000000, 5000000, 30),
#         'large_mid_sell': np.random.uniform(1000000, 5000000, 30),
#         'small_buy': np.random.uniform(100000, 500000, 30),
#         'small_sell': np.random.uniform(100000, 500000, 30),
#         'returns': np.random.uniform(-0.05, 0.05, 30)
#     })
    
#     # 设置日期为索引
#     sample_data.set_index('date', inplace=True)
    
#     # 计算ACT因子
#     results = calculate_act_factors(sample_data)
    
#     # 打印结果
#     print("\n高收益日正向ACT因子:")
#     print(results['high_return_act_positive'].dropna().tail())
#     print("\n低收益日负向ACT因子:")
#     print(results['low_return_act_negative'].dropna().tail())

# if __name__ == "__main__":
#     example_usage()