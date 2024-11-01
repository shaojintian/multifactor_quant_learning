import pandas as pd
import numpy as np

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
    
    # 2. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # 3. 正交化处理
    orthogonal_factors = pd.DataFrame(
        np.dot(factors, eigenvectors),
        index=factors.index,
        columns=[f'orthogonal_factor_{i}' for i in range(len(factors.columns))]
    )
    
    return orthogonal_factors


def process_orthogonalized_factors(orthogonal_factors: pd.DataFrame) -> pd.DataFrame:
    """
    处理正交化后的因子值
    """
    # 方法1：直接标准化后再截断
    def standardize_and_clip(series):
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


def process_multi_factors(factors: pd.DataFrame) -> pd.DataFrame:
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