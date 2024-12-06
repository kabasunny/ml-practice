# cycle_theory\fft_cycle_analysis_1.py

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os


def fft_cycle_analysis(symbol, start_date, end_date):
    # 1. データ取得
    data = yf.download(symbol, start=start_date, end=end_date)
    close_prices = data["Close"].values
    n = len(close_prices)
    print("取得したデータの終値:", close_prices)
    print("データポイント数:", n)

    # 2. データの前処理
    detrended_prices = close_prices - np.mean(
        close_prices
    )  # 平均を引いてトレンドを除去
    print("平均を引いた後のデータ:", detrended_prices)

    # 3. FFT（高速フーリエ変換）
    fft_result = np.fft.fft(detrended_prices)  # FFTを計算
    fft_freq = np.fft.fftfreq(n)  # 周波数成分
    print("FFTの結果:", fft_result)
    print("周波数成分:", fft_freq)

    # 4. 振幅スペクトルの計算
    fft_amplitude = np.abs(fft_result)[: n // 2]  # 振幅（スペクトルの前半部）
    fft_period = 1 / fft_freq[: n // 2]  # 周期（スペクトルの前半部）
    print("振幅スペクトル:", fft_amplitude)
    print("周期スペクトル:", fft_period)

    # 5. 意味のある周期の特定（最も振幅が大きい成分）
    valid_idx = np.where(fft_period > 0)  # 正の周期のみ
    dominant_periods = fft_period[valid_idx][
        np.argsort(-fft_amplitude[valid_idx])[:5]
    ]  # 上位5つの周期
    print("上位5つの周期:", dominant_periods)

    # 6. 結果の表示
    print(f"銘柄: {symbol}")
    print(f"上位の周期 (日): {dominant_periods}")

    # ディレクトリの作成
    directory = (
        f"chart1_{symbol}_{start_date.replace('-', '')}-{end_date.replace('-', '')}"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 7. detrended_pricesのチャート描画と保存
    plt.figure(figsize=(12, 6))
    plt.plot(detrended_prices, label="Detrended Prices")
    plt.title(f"Detrended Prices for {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, "detrended_prices.png"))
    plt.close()

    # 8. FFTのチャート描画と保存
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
    plt.savefig(os.path.join(directory, "fft_analysis.png"))
    plt.close()


# 使用例
fft_cycle_analysis(
    symbol="^TOPX",  # 日経平均株価のティッカーシンボル ^TOPX ^N225
    start_date="2013-01-01",  # 取得開始日
    end_date="2023-12-01",  # 取得終了日
)
