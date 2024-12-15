# data_processing\detrend_prices.py
import numpy as np
import pandas as pd

# データの前処理（線形トレンド除去）関数
def detrend_prices(prices, remove_trend=True):
    if remove_trend:
        n = len(prices)

        # 線形トレンドの計算
        slope = (prices.iloc[-1] - prices.iloc[0]) / n
        trend = np.arange(n) * slope + prices.iloc[0]

        # トレンドを除去したデータ
        detrended_prices = prices - trend

        # 結果を pandas.Series に変換
        detrended_prices_series = pd.Series(detrended_prices, index=prices.index)

    else:
        # トレンド除去を行わない場合
        detrended_prices = prices

        # 結果を pandas.Series に変換
        detrended_prices_series = pd.Series(detrended_prices, index=prices.index)

    return detrended_prices_series
