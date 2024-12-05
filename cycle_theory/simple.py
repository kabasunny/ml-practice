# cycle_theory\cycle_theory_simple.py
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def extract_cycle(symbol, start_date, end_date):
    # 1. データの取得
    data = yf.download(symbol, start=start_date, end=end_date)
    close_prices = data["Close"]

    # インデックスを日付に設定
    data["Date"] = data.index
    data = data.reset_index(drop=True)

    # 2. ピークと谷の検出
    peaks, _ = find_peaks(close_prices)  # 高値のピーク
    troughs, _ = find_peaks(-close_prices)  # 安値の谷

    # 3. サイクルの長さを計算
    peak_intervals = np.diff(peaks) if len(peaks) > 1 else []
    trough_intervals = np.diff(troughs) if len(troughs) > 1 else []

    # 平均サイクル長を計算
    avg_peak_cycle = np.mean(peak_intervals) if len(peak_intervals) > 0 else None
    avg_trough_cycle = np.mean(trough_intervals) if len(trough_intervals) > 0 else None

    # 4. 結果の表示
    print(f"銘柄: {symbol}")
    print(f"高値サイクルの平均: {avg_peak_cycle} 日")
    print(f"安値サイクルの平均: {avg_trough_cycle} 日")

    # 5. グラフの描画
    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], close_prices, label="Close Price", color="blue")
    plt.scatter(
        data["Date"].iloc[peaks],
        close_prices.iloc[peaks],
        color="red",
        label="Peaks (High)",
    )
    plt.scatter(
        data["Date"].iloc[troughs],
        close_prices.iloc[troughs],
        color="green",
        label="Troughs (Low)",
    )
    plt.title(f"Cycle Analysis for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# 使用例
extract_cycle(
    symbol="AAPL",  # 銘柄コード（例: Apple Inc.）
    start_date="2023-11-01",  # 取得開始日
    end_date="2023-12-31",  # 取得終了日
)
