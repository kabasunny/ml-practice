# data_processing\detrend_prices.py
import numpy as np
import pandas as pd

# pandas.Series を戻す
# データの前処理（終値のみ線形トレンド除去）関数
def detrend_prices(prices, remove_trend=True):
    if remove_trend:
        n = len(prices)
        prices = prices['Close']  # とりあえず'Close' 列を使用

        # 線形トレンドの計算
        slope = (prices.iloc[-1] - prices.iloc[0]) / (n - 1)
        trend = np.arange(n) * slope + prices.iloc[0]

        # トレンドを除去したデータ
        detrended_prices = prices - trend

        # 結果を pandas.Series に変換
        detrended_prices_series = pd.Series(detrended_prices, index=prices.index)

    else:
        # トレンド除去を行わない場合
        prices = prices['Close']  # 'Close' 列を使用
        detrended_prices_series = pd.Series(prices, index=prices.index)

    return detrended_prices_series  # pandas.Series を戻す

# Detrended Prices:
# Date
# 2023-01-04     0.000000
# 2023-01-05     5.282520
# 2023-01-06    19.565041
# 2023-01-10    18.347561
# 2023-01-11    25.630081
#                 ...
# 2023-12-25   -37.412602
# 2023-12-26   -36.630081
# 2023-12-27     2.152439
# 2023-12-28   -28.065041
# 2023-12-29     3.217480
# Name: Close, Length: 246, dtype: float64
