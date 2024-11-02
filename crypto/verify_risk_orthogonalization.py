import pandas as pd
import numpy as np
from model.random_forest import combine_factors_nonlinear
import seaborn as sns
from matplotlib import pyplot as plt
from util.norm import normalize_factor
from util.decorator import print_variable_shapes

def risk_orthogonalization(factors: pd.DataFrame) -> pd.DataFrame:
    """
    风险正交化处理
    
    主要步骤：
    1. 计算因子相关性矩阵
    2. 进行特征值分解
    3. 重构正交因子
    """
    # 1. 计算相关性矩阵
    corr_matrix = factors.corr()

    #1.1 可视化因子相关性
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(" 可视化正交化前的因子相关性")
    plt.show()
    
    # 2. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # 3. 正交化处理
    orthogonal_factors = pd.DataFrame(
        np.dot(factors, eigenvectors),
        index=factors.index,
        columns=[f'orthogonal_factor_{i}' for i in range(len(factors.columns))]
    )
    #4. 可视化正交化前后的因子相关性
    plt.figure(figsize=(10, 8))
    sns.heatmap(orthogonal_factors.corr(), annot=True, cmap='coolwarm')
    plt.title(" 可视化正交化后的因子相关性")
    plt.show()
    return orthogonal_factors


def process_orthogonalized_factors(orthogonal_factors: pd.DataFrame) -> pd.DataFrame:
    """
    处理正交化后的因子值
    """
    # 方法1：直接标准化后再截断
    def standardize_and_clip(series):
        if series.lengthth < 2000:
            raise ValueError("process_orthogonalized_factors Series length is less than 2000")
        normalized = (series - series.rolling(window=2000).mean()) / series.rolling(window=2000).std()
        return normalized.clip(-3, 3)
    
    # 方法2：基于百分位数的缩放
    def percentile_scale(series):
        upper = series.quantile(0.9973)  # 3σ对应的概率约为99.73%
        lower = series.quantile(0.0027)
        scaled = 6 * (series - lower) / (upper - lower) - 3
        return scaled.clip(-3, 3)
    
    # 方法3：保持相对比例的缩放
    def proportional_scale(series):
        max_abs = max(abs(series.max()), abs(series.min()))
        return (series * 3 / max_abs).clip(-3, 3)
    
    processed_factors = pd.DataFrame()
    
    # 对每个正交化后的因子进行处理
    for col in orthogonal_factors.columns:
        # 选择其中一种方法应用
        processed_factors[col] = standardize_and_clip(orthogonal_factors[col])
        
    return processed_factors


def process_multi_factors_linear(factors: pd.DataFrame) -> pd.DataFrame:
    """
    完整的多因子处理流程
    """
    # 1. 首先进行风险正交化
    orthogonal_factors = risk_orthogonalization(factors)
    
    # 2. 处理正交化后的因子值
    processed_factors = process_orthogonalized_factors(orthogonal_factors)
    
    # 3. 计算组合权重（可选）
    def calculate_weights(processed_factors):
        # 等权重
        weights = np.ones(len(processed_factors.columns)) / len(processed_factors.columns)
        return weights
    
    # 4. 合成最终因子（可选）
    weights = calculate_weights(processed_factors)
    final_factor = (processed_factors * weights).sum(axis=1)
    final_factor.name = u'final_factor which combines all factors'
    
    return processed_factors, final_factor


def process_multi_factors_nonlinear(factors_df:pd.DataFrame, returns:pd.Series ,model_type='xgboost') -> pd.DataFrame:
    """
    完整的多因子处理流程
    """
    # 1. 首先进行风险正交化
    orthogonal_factors = risk_orthogonalization(factors_df)
    orthogonal_factors.name = u'orthogonal_factor'
    #print(orthogonal_factors.describe())
    
    # 2. 处理正交化后的因子值
    processed_factors = process_orthogonalized_factors(orthogonal_factors)
    processed_factors.name = u'processed_factors'
    #print(processed_factors.describe())
    #processed_factors.to_csv('factor_test_data/processed_factors.csv')
    
    # 3. model
    final_factor ,model = combine_factors_nonlinear(factors_df=processed_factors,returns=returns,model_type=model_type)
    #print(final_factor.shape)

    # 4. 最终因子标准化
    final_factor = normalize_factor(final_factor)
    final_factor.name = u'normalzied combined_factor '
    
    return processed_factors, final_factor