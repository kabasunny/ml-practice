import numpy as np
from scipy.signal import find_peaks

# サイクル検出関数
def detect_cycles(close_prices):
    # 終値のデータからピーク（高値）を検出
    peaks, _ = find_peaks(close_prices)
    
    # 終値のデータから谷（安値）を検出
    troughs, _ = find_peaks(-close_prices)

    # ピーク間の間隔を計算
    peak_intervals = np.diff(peaks)
    
    # 谷間の間隔を計算
    trough_intervals = np.diff(troughs)

    # 一番大きい値と一番小さい値を除外して平均を計算
    if len(peak_intervals) > 2:
        peak_intervals = np.sort(peak_intervals)[1:-1]  # 一番小さい値と一番大きい値を除外
    if len(trough_intervals) > 2:
        trough_intervals = np.sort(trough_intervals)[1:-1]  # 一番小さい値と一番大きい値を除外

    # ピーク間の平均サイクルを少数第一位まで計算
    avg_peak_cycle = round(np.mean(peak_intervals), 1) if len(peak_intervals) > 0 else None
    
    # 谷間の平均サイクルを少数第一位まで計算
    avg_trough_cycle = round(np.mean(trough_intervals), 1) if len(trough_intervals) > 0 else None

    # ピーク、谷、平均ピークサイクル、平均谷サイクルを返す
    return peaks, troughs, avg_peak_cycle, avg_trough_cycle
