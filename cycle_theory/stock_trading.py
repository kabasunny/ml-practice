# cycle_theory_stock_trading.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.filters.hp_filter import hpfilter

# pip install statsmodels


# 1. データ収集と前処理
def preprocess_data(price_data):
    # 平滑化（ハドリック-プレスコットフィルタを適用）
    cycle, trend = hpfilter(price_data["close"], lamb=1600)
    return cycle


# 2. サイクルの特定
def detect_cycles(price_data):
    # 価格データからピークを検出
    peaks, _ = find_peaks(price_data["close"])
    # 価格データからトラフ（谷）を検出
    troughs, _ = find_peaks(-price_data["close"])

    # 各サイクルの長さを計算
    cycle_lengths = np.diff(peaks)
    # 平均サイクル長を計算
    avg_cycle_length = np.mean(cycle_lengths)
    return peaks, troughs, avg_cycle_length


# 3. サイクルの分類
def classify_cycle(peaks, troughs):
    if len(peaks) < 2:
        return "Insufficient data"  # ピークが2つ未満の場合、データが不十分
    durations = np.diff(peaks)  # 各サイクルの期間を計算
    if durations[-1] > np.mean(durations):
        return "Right Translation"  # 最新サイクルが平均より長い場合、右翻訳
    else:
        return "Left Translation"  # 最新サイクルが平均以下の場合、左翻訳


# 4. サイクル予測
def predict_next_cycle(peaks, avg_cycle_length):
    # 次のピークの予測位置を計算
    next_peak = peaks[-1] + avg_cycle_length
    return next_peak


# 5. リスク管理（仮のロジック）
def risk_management(entry_point, stop_loss_ratio=0.05):
    # 損切りポイントを設定
    stop_loss = entry_point * (1 - stop_loss_ratio)
    return stop_loss


# メイン処理
price_data = pd.DataFrame({"close": [100, 102, 101, 105, 110, 108, 112, 120]})
cycle_data = preprocess_data(price_data)  # データを平滑化
peaks, troughs, avg_cycle_length = detect_cycles(price_data)  # サイクルを特定
classification = classify_cycle(peaks, troughs)  # サイクルを分類
next_peak_prediction = predict_next_cycle(peaks, avg_cycle_length)  # 次のサイクルを予測
stop_loss = risk_management(price_data["close"].iloc[-1])  # 損切りポイントを設定

# 結果を表示
print(f"サイクル分類: {classification}")
print(f"次のサイクルの予測: {next_peak_prediction}")
print(f"損切りポイント: {stop_loss}")
