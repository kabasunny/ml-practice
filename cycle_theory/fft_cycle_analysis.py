import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def fft_cycle_analysis(symbol, start_date, end_date):
    # 1. データ取得
    data = yf.download(symbol, start=start_date, end=end_date)
    close_prices = data["Close"].values
    n = len(close_prices)

    # 2. データの前処理
    detrended_prices = close_prices - np.mean(
        close_prices
    )  # 平均を引いてトレンドを除去

    # 3. FFT（高速フーリエ変換）
    fft_result = np.fft.fft(detrended_prices)  # FFTを計算
    fft_freq = np.fft.fftfreq(n)  # 周波数成分

    # 4. 振幅スペクトルの計算
    fft_amplitude = np.abs(fft_result)[: n // 2]  # 振幅（スペクトルの前半部）
    fft_period = 1 / fft_freq[: n // 2]  # 周期（スペクトルの前半部）

    # 5. 意味のある周期の特定（最も振幅が大きい成分）
    valid_idx = np.where(fft_period > 0)  # 正の周期のみ
    dominant_periods = fft_period[valid_idx][
        np.argsort(-fft_amplitude[valid_idx])[:5]
    ]  # 上位5つの周期

    # 6. 結果の表示
    print(f"銘柄: {symbol}")
    print(f"上位の周期 (日): {dominant_periods}")

    # 7. グラフの描画
    plt.figure(figsize=(12, 6))
    plt.plot(
        fft_period[valid_idx], fft_amplitude[valid_idx], label="Amplitude Spectrum"
    )
    plt.xscale("log")  # X軸を対数スケールに変更
    plt.title(f"FFT Cycle Analysis for {symbol}")
    plt.xlabel("Period (Days)")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


# 使用例
fft_cycle_analysis(
    symbol="AAPL",  # 銘柄コード（例: Apple Inc.）
    start_date="2013-01-01",  # 取得開始日
    end_date="2023-12-01",  # 取得終了日
)
