import pandas as pd
import talib as ta
from norm import normalize_factor

def bolling_band_factor_generator(z:pd.DataFrame) ->pd.Series:
    # input is pd.Series bolling band
    upper, middle, lower = ta.BBANDS(z['close'], timeperiod=100, nbdevup=2, nbdevdn=2, matype=0)
    # Calculate the position of the price within the Bollinger Bands
    bolling_band_factor = (z['close'] - lower) / (upper - lower)
    z['bolling_band_factor'] = bolling_band_factor

    return normalize_factor(z['bolling_band_factor'])  