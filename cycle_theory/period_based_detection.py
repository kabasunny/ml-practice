import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yfinance as yf
import matplotlib.pyplot as plt


# データの取得関数
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


# サイクル検出関数
def detect_cycles_by_period(close_prices, period):
    peaks = []
    troughs = []
    for start in range(0, len(close_prices), period):
        end = start + period
        if end > len(close_prices):
            end = len(close_prices)
        segment = close_prices[start:end]
        if len(segment) == 0:
            continue
        peak, _ = find_peaks(segment)
        trough, _ = find_peaks(-segment)
        if len(peak) > 0:
            peaks.append(peak[0] + start)  # 相対インデックスを絶対インデックスに変換
        if len(trough) > 0:
            troughs.append(
                trough[0] + start
            )  # 相対インデックスを絶対インデックスに変換

    peak_intervals = np.diff(peaks)
    trough_intervals = np.diff(troughs)
    avg_peak_cycle = np.mean(peak_intervals) if len(peak_intervals) > 0 else None
    avg_trough_cycle = np.mean(trough_intervals) if len(trough_intervals) > 0 else None

    return peaks, troughs, avg_peak_cycle, avg_trough_cycle


# グラフ描画関数
def plot_cycles(data, peaks, troughs, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label="Close Price", color="blue")
    plt.scatter(data.index[peaks], data.iloc[peaks], color="red", label="Peaks")
    plt.scatter(data.index[troughs], data.iloc[troughs], color="green", label="Troughs")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


# メイン処理
symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2023-12-01"
period = 10  # 10日間ごとにピークを検出

# データの取得
data = fetch_data(symbol, start_date, end_date)
close_prices = data["Close"]

# サイクルの検出
peaks, troughs, avg_peak_cycle, avg_trough_cycle = detect_cycles_by_period(
    close_prices, period
)
print(f"平均ピークサイクル: {avg_peak_cycle} 日")
print(f"平均谷サイクル: {avg_trough_cycle} 日")

# チャートの描画
plot_cycles(
    close_prices, peaks, troughs, f"Cycle Analysis for {symbol} (Period: {period} days)"
)
