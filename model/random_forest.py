import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def combine_factors_nonlinear(factors_df:pd.DataFrame, returns:pd.Series ,model_type='rf', n_splits=5):
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
    predictions = np.zeros_like(y)
    feature_importance = pd.Series(0, index=X.columns)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测并存储结果
        predictions[test_idx] = model.predict(X_test)
        
        # 累积特征重要性
        if model_type == 'rf':
            feature_importance += pd.Series(
                model.feature_importances_,
                index=X.columns
            )
        else:
            feature_importance += pd.Series(
                model.feature_importances_,
                index=X.columns
            )
    
    # 平均特征重要性
    feature_importance /= n_splits
    
    # 创建最终预测作为组合因子
    final_factor = pd.Series(predictions, index=y.index)
    final_factor.name = 'combined_factor'
    
    # 打印模型评估结果
    r2 = r2_score(y, predictions)
    print(f"R-squared score: {r2:.4f}")
    print("\nFeature Importance:")
    print(feature_importance.sort_values(ascending=False))
    
    return final_factor, model

# # 使用示例：
# # 假设 processed_factors 是之前处理好的因子DataFrame，ret 是未来收益率
# combined_factor, model = combine_factors_nonlinear(
#     processed_factors,
#     ret,
#     model_type='xgb',
#     n_splits=5
# )

# # 可视化组合因子的预测效果
# plt.figure(figsize=(12, 6))
# plt.scatter(combined_factor, ret, alpha=0.5)
# plt.xlabel('Combined Factor Prediction')
# plt.ylabel('Actual Returns')
# plt.title('Combined Factor vs Actual Returns')
# plt.grid(True)
# plt.show()

# # 使用组合因子计算持仓
# pos = combined_factor
# net_values = cal_net_values(pos, ret)

# plt.figure(figsize=(12, 6))
# plt.plot(net_values.values)