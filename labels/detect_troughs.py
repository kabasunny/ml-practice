import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mode

# トラフ（谷）検出関数
def detect_troughs(prices):
    # 終値のデータから谷（安値）を検出
    troughs, _ = find_peaks(-prices)

    # 谷間の間隔を計算
    trough_intervals = np.diff(troughs)

    # 谷間の中央値を計算
    median_trough_cycle = np.median(trough_intervals) if len(trough_intervals) > 0 else None

    # 谷間の平均サイクルを少数第一位まで計算
    avg_trough_cycle = round(np.mean(trough_intervals), 1) if len(trough_intervals) > 0 else None

    # 谷間の平均誤差のパーセンテージを少数第一位まで計算
    mean_absolute_error_trough = (
        round(np.mean(np.abs(trough_intervals - avg_trough_cycle)) / avg_trough_cycle * 100, 1)
        if avg_trough_cycle
        else None
    )

    # 谷間の最頻値を安全に計算
    mode_trough_cycle = None
    if len(trough_intervals) > 0:
        mode_result = mode(trough_intervals)
        # mode_result.modeが配列かどうかをチェックし、適切に処理する
        if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
            mode_trough_cycle = mode_result.mode[0]
        else:
            mode_trough_cycle = mode_result.mode

    # 谷、平均谷サイクル、中央値、平均誤差、最頻値を返す
    return (
        troughs,  # インデックス
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
        mode_trough_cycle,
    )

