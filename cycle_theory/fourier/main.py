# cycle_theory\fourier\main.py

import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import modules from the project root
from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from fft_analysis import fft_analysis
from plot_fft import plot_fft
import pandas as pd


# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # データの取得
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas

    # 前処理（線形トレンド除去）
    detrended_prices = detrend_prices(data)

    # FFTによるサイクル解析（日足）
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("上位5つの周期 (日足):", dominant_periods)

    # FFTスペクトルのプロット（日足）
    # plot_fft(fft_period, fft_amplitude, "FFT Analysis (Log Scale) - Daily")

    # データのリサンプリング（週足）
    weekly_data = detrended_prices.resample(
        "W"
    ).ffill()  # 週足にリサンプリングし、前週の値を埋める（pandas を必要とする）

    # FFTによるサイクル解析（週足）
    fft_period, fft_amplitude, dominant_periods = fft_analysis(weekly_data.values)
    print("上位5つの周期 (週足):", dominant_periods)

    # FFTスペクトルのプロット（週足）
    # plot_fft(fft_period, fft_amplitude, "FFT Analysis (Log Scale) - Weekly")

    # データのリサンプリング（月足）
    monthly_data = detrended_prices.resample(
        "ME"
    ).ffill()  # 月足にリサンプリングし、前月の値を埋める（pandas を必要とする）

    # FFTによるサイクル解析（月足）
    fft_period, fft_amplitude, dominant_periods = fft_analysis(monthly_data.values)
    print("上位5つの周期 (月足):", dominant_periods)

    # FFTスペクトルのプロット（月足）
    # plot_fft(fft_period, fft_amplitude, "FFT Analysis (Log Scale) - Monthly")


if __name__ == "__main__":
    main()
