# cycle_theory\monthly_analysis.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yfinance as yf
import matplotlib.pyplot as plt


# データの取得関数
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


# データのリサンプリング関数（月足）
def resample_monthly(data):
    return data.resample("M").ffill()  # 月足にリサンプリングし、前月の値を埋める


# サイクル検出関数
def detect_cycles(close_prices):
    peaks, _ = find_peaks(close_prices)
    troughs, _ = find_peaks(-close_prices)

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
start_date = "2013-01-01"
end_date = "2023-12-01"

# データの取得
data = fetch_data(symbol, start_date, end_date)

# データのリサンプリング（月足）
monthly_data = resample_monthly(data["Close"])

# サイクルの検出
peaks, troughs, avg_peak_cycle, avg_trough_cycle = detect_cycles(monthly_data)
print(f"平均ピークサイクル: {avg_peak_cycle} ヶ月")
print(f"平均谷サイクル: {avg_trough_cycle} ヶ月")

# チャートの描画
plot_cycles(monthly_data, peaks, troughs, f"Monthly Cycle Analysis for {symbol}")
