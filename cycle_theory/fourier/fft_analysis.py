# cycle_theory\fourier\fft_analysis.py
import numpy as np


# フーリエ変換を用いたサイクル解析関数
def fft_analysis(prices):
    n = len(prices)  # 終値データの長さを取得

    # FFT（高速フーリエ変換）
    fft_result = np.fft.fft(prices)  # FFTを実行
    fft_amplitude = np.abs(fft_result)[: n // 2]  # 振幅スペクトルを計算（前半部分のみ）
    fft_freq = np.fft.fftfreq(n, d=1)[: n // 2]  # 周波数を計算（前半部分のみ）

    # 周期計算
    non_zero_idx = fft_freq > 0  # 周波数がゼロでないインデックスを取得
    fft_period = 1 / fft_freq[non_zero_idx]  # 周期を計算
    fft_amplitude = fft_amplitude[
        non_zero_idx
    ]  # 振幅スペクトルを対応する部分に絞り込み

    # 結果を少数第一位までに丸める
    fft_period = np.round(fft_period, 1)  # 周期を少数第一位までに丸める
    fft_amplitude = np.round(fft_amplitude, 1)  # 振幅スペクトルを少数第一位までに丸める

    # # 最大値を除外
    # if len(fft_amplitude) > 1:
    #     max_idx = np.argmax(fft_amplitude)  # 最大値のインデックスを取得
    #     excluded_value = fft_period[max_idx]  # 除外する最大周期の値を取得
    #     fft_amplitude = np.delete(
    #         fft_amplitude, max_idx
    #     )  # 最大値を振幅スペクトルから除去
    #     fft_period = np.delete(
    #         fft_period, max_idx
    #     )  # 信頼性が低いと考え、最大周期を周期リストから除去
    #     print(f"excluded_fft_period : {excluded_value}")  # 除外した値をアナウンス

    dominant_periods = fft_period[
        np.argsort(-fft_amplitude)[:10]
    ]  # 上位5つの支配的な周期を取得

    return fft_period, fft_amplitude, dominant_periods  # 計算結果を返す
