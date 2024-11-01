# %%
import pandas as pd
# 3. 标准化因子
def normalize_factor(factor:pd.Series,window:int=2000) -> pd.Series:
    return ((factor - factor.rolling(window=window).mean()) / factor.rolling(window=window).std()).clip(-3,3)
