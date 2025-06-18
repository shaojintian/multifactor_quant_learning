

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