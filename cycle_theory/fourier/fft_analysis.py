import numpy as np

# フーリエ変換を用いたサイクル解析関数
def fft_analysis(close_prices):
    n = len(close_prices)

    # データの前処理（線形トレンド除去）
    trend = np.linspace(close_prices[0], close_prices[-1], n)
    detrended_prices = close_prices - trend

    # FFT（高速フーリエ変換）
    fft_result = np.fft.fft(detrended_prices)
    fft_amplitude = np.abs(fft_result)[: n // 2]
    fft_freq = np.fft.fftfreq(n, d=1)[: n // 2]

    # 周期計算
    non_zero_idx = fft_freq > 0
    fft_period = 1 / fft_freq[non_zero_idx]
    fft_amplitude = fft_amplitude[non_zero_idx]  # non_zero_idx で絞り込んで同じ長さに

    dominant_periods = fft_period[np.argsort(-fft_amplitude)[:5]]

    return fft_period, fft_amplitude, dominant_periods
