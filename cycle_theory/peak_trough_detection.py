import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# サンプルデータ
close_prices = pd.Series([100, 102, 101, 105, 110, 108, 112, 120])

# ピークと谷の検出
peaks, _ = find_peaks(close_prices)
troughs, _ = find_peaks(-close_prices)

# サイクルの長さを計算
peak_intervals = np.diff(peaks)  # ピーク間の期間
trough_intervals = np.diff(troughs)  # 谷間の期間

# 平均サイクル長を計算
avg_peak_cycle = np.mean(peak_intervals) if len(peak_intervals) > 0 else None
avg_trough_cycle = np.mean(trough_intervals) if len(trough_intervals) > 0 else None

# 結果の表示
print(f"ピーク: {peaks} ")
print(f"ピーク間の平均サイクル: {avg_peak_cycle} 日")
print(f"谷: {troughs} ")
print(f"谷間の平均サイクル: {avg_trough_cycle} 日")

# チャートの描画
plt.figure(figsize=(10, 6))
plt.plot(close_prices, label="Close Price", color="blue")
plt.scatter(peaks, close_prices[peaks], color="red", label="Peaks")
plt.scatter(troughs, close_prices[troughs], color="green", label="Troughs")
plt.title("Cycle Analysis")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.show()
