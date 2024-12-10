# cycle_theory\peak_trough\detect_cycles.py
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mode


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

    # # 一番大きい値と一番小さい値を除外して平均を計算
    # if len(peak_intervals) > 2:
    #     sorted_peak_intervals = np.sort(peak_intervals)
    #     excluded_peak_intervals = (sorted_peak_intervals[0], sorted_peak_intervals[-1])
    #     peak_intervals = sorted_peak_intervals[1:-1]  # 一番小さい値と一番大きい値を除外

    # if len(trough_intervals) > 2:
    #     sorted_trough_intervals = np.sort(trough_intervals)
    #     excluded_trough_intervals = (
    #         sorted_trough_intervals[0],
    #         sorted_trough_intervals[-1],
    #     )
    #     trough_intervals = sorted_trough_intervals[
    #         1:-1
    #     ]  # 一番小さい値と一番大きい値を除外
    #     print(
    #         f"excluded_cycles : [peak: min: {excluded_peak_intervals[0]}, max: {excluded_peak_intervals[1]}] [trough: min: {excluded_trough_intervals[0]}, max: {excluded_trough_intervals[1]}]"
    #     )

    # ピーク間の中央値を計算
    median_peak_cycle = np.median(peak_intervals) if len(peak_intervals) > 0 else None

    # ピーク間の平均サイクルを少数第一位まで計算
    avg_peak_cycle = (
        round(np.mean(peak_intervals), 1) if len(peak_intervals) > 0 else None
    )

    # ピーク間の平均誤差のパーセンテージを少数第一位まで計算
    mean_absolute_error_peak = (
        round(
            np.mean(np.abs(peak_intervals - avg_peak_cycle)) / avg_peak_cycle * 100, 1
        )
        if avg_peak_cycle
        else None
    )

    # ピーク間の最頻値を安全に計算
    mode_peak_cycle = None
    if len(peak_intervals) > 0:
        mode_result = mode(peak_intervals)
        # mode_result.modeが配列かどうかをチェックし、適切に処理する
        if isinstance(mode_result.mode, np.ndarray) and mode_result.mode.size > 0:
            mode_peak_cycle = mode_result.mode[0]
        else:
            mode_peak_cycle = mode_result.mode

    # 谷間の中央値を計算
    median_trough_cycle = (
        np.median(trough_intervals) if len(trough_intervals) > 0 else None
    )

    # 谷間の平均サイクルを少数第一位まで計算
    avg_trough_cycle = (
        round(np.mean(trough_intervals), 1) if len(trough_intervals) > 0 else None
    )

    # 谷間の平均誤差のパーセンテージを少数第一位まで計算
    mean_absolute_error_trough = (
        round(
            np.mean(np.abs(trough_intervals - avg_trough_cycle))
            / avg_trough_cycle
            * 100,
            1,
        )
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

    # ピーク、谷、平均ピークサイクル、中央値、平均誤差、最頻値、平均谷サイクル、中央値、平均誤差、最頻値を返す
    return (
        peaks,  # インデックス
        troughs,  # インデックス
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        mode_peak_cycle,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
        mode_trough_cycle,
    )
