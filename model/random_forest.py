import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def combine_factors_nonlinear(factors_df: pd.DataFrame, returns: pd.Series, model_type='rf', n_splits=5):
    """
    使用非线性模型组合多因子
    
    Parameters:
    -----------
    factors_df: DataFrame
        多个因子值的数据框
    returns: Series
        未来收益率
    model_type: str
        'rf' for Random Forest, 'xgb' for XGBoost
    n_splits: int
        时间序列交叉验证的折数
    
    Returns:
    --------
    final_factor: Series
        组合后的因子值
    model: object
        训练好的模型
    """
    
    # 准备特征和标签
    X = factors_df.copy()
    y = returns.copy()
    
    # 去除包含 NaN 的行
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # 填充 NaN 值为上一个有效值
    X = X.fillna(method='ffill')
    y = y.fillna(method='ffill')

    # 创建一个与原始 y 相同长度的预测数组
    predictions = np.zeros(len(returns))  # 初始化为零

    # 选择模型
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=100,
            random_state=42
        )
    else:  # xgboost
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=100,
            random_state=42
        )
    
    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测并存储结果
        predictions[test_idx] = model.predict(X_test)

    # 对齐索引，确保 final_factor 的索引与 returns 一致
    final_factor = pd.Series(predictions, index=returns.index)
    final_factor = final_factor[returns.index]  # 只保留与 returns 对应的索引
    final_factor.name = 'combined_factor'
    
    # 打印模型评估结果
    r2 = r2_score(returns, final_factor)
    print(f"R-squared score: {r2:.4f}")
    
    return final_factor, model

# 使用示例：
# combined_factor, model = combine_factors_nonlinear(
#     processed_factors,
#     ret,
#     model_type='xgb',
#     n_splits=5
# )