# cycle_theory\peak_trough\main.py

import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from detect_cycles import detect_cycles
from plot_cycles import plot_cycles
import pandas as pd

def main():
    symbol = "7203.T"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # データの取得（日足）
    daily_data, weekly_data, monthly_data = fetch_stock_data(symbol, start_date, end_date)
    
    # 前処理（線形トレンド除去）
    detrended_daily_data = detrend_prices(daily_data['Close'], remove_trend=True)
    detrended_weekly_data = detrend_prices(weekly_data['Close'], remove_trend=True)
    detrended_monthly_data = detrend_prices(monthly_data['Close'], remove_trend=True)

    # detrended_daily_dataとdetrended_weekly_dataの要素の値が同じものを抽出
    common_elements = detrended_daily_data[detrended_daily_data.isin(detrended_weekly_data)]

    # 抽出した要素を表示
    print("Common elements between detrended_daily_data and detrended_weekly_data:")
    print(common_elements)
    
    print(f"symbol: {symbol} , start_date: {start_date} , end_date: {end_date}")
    print("----------------    日足    ----------------")

    # サイクルの検出（日足）
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        mode_peak_cycle,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
        mode_trough_cycle,
    ) = detect_cycles(detrended_daily_data)  # Close列ではなくdetrended_prices全体を渡す
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # チャートの描画（日足）
    plot_cycles(
        detrended_daily_data, peaks, troughs, f"Daily Cycle Analysis for {symbol}"
    )

    print("----------------    週足    ----------------")

    # データのリサンプリング（週足）
    # weekly_data_resampled = weekly_data.resample('W').last()

    # サイクルの検出（週足）
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        mode_peak_cycle,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
        mode_trough_cycle,
    ) = detect_cycles(detrended_weekly_data)  # Close列を渡す
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # チャートの描画（週足）
    plot_cycles(
        detrended_weekly_data, peaks, troughs, f"Weekly Cycle Analysis for {symbol}"
    )

    print("----------------    月足    ----------------")

    # データのリサンプリング（月足）
    # monthly_data_resampled = monthly_data.resample('ME').last()

    # サイクルの検出（月足）
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        mode_peak_cycle,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
        mode_trough_cycle,
    ) = detect_cycles(detrended_monthly_data)  # Close列を渡す
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # チャートの描画（月足）
    plot_cycles(
        detrended_monthly_data, peaks, troughs, f"Monthly Cycle Analysis for {symbol}"
    )

if __name__ == "__main__":
    main()
