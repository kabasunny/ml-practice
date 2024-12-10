# cycle_theory\peak_trough\main.py

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
from detect_cycles import detect_cycles
from plot_cycles import plot_cycles


# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # データの取得（日足）
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas

    # 前処理（線形トレンド除去）
    detrended_prices = detrend_prices(data)

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
    ) = detect_cycles(detrended_prices)
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # # チャートの描画（日足）
    plot_cycles(
        detrended_prices, peaks, troughs, f"Daily Cycle Analysis for {symbol}"
    )  # pandas を必要とする

    print("----------------    週足    ----------------")

    # データのリサンプリング関数（週足）
    weekly_data = detrended_prices.resample(
        "W"
    ).ffill()  # 週足にリサンプリングし、前週の値を埋める（pandas を必要とする）

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
    ) = detect_cycles(weekly_data)
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # # チャートの描画（週足）
    plot_cycles(
        weekly_data, peaks, troughs, f"Weekly Cycle Analysis for {symbol}"
    )  # pandas を必要とする

    print("----------------    月足    ----------------")

    # データのリサンプリング関数（月足）
    monthly_data = detrended_prices.resample(
        "ME"
    ).ffill()  # 月足にリサンプリングし、前月の値を埋める（pandas を必要とする）

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
    ) = detect_cycles(monthly_data)
    print(
        f"peak_cycles    : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles  : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # # チャートの描画（月足）
    plot_cycles(
        monthly_data, peaks, troughs, f"Monthly Cycle Analysis for {symbol}"
    )  # pandas を必要とする


if __name__ == "__main__":
    main()
