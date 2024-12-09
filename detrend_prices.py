import numpy as np
import pandas as pd

# データの前処理（線形トレンド除去）関数
def detrend_prices(data):
    close_prices = data["Close"].values  # Close 列を numpy 配列に変換

    n = len(close_prices)
    
    # 線形トレンドの計算
    slope = (close_prices[-1] - close_prices[0]) / n
    trend = np.arange(n) * slope + close_prices[0]
    
    # トレンドを除去したデータ
    detrended_prices = close_prices - trend
    
    # 結果を pandas.Series に変換
    detrended_prices_series = pd.Series(detrended_prices, index=data.index)
    
    return detrended_prices_series
