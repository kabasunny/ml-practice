from fetch_stock_data import fetch_stock_data
from detrend_prices import detrend_prices
from cycle_theory.peak_trough.detect_cycles import detect_cycles
from cycle_theory.fourier.fft_analysis import fft_analysis


# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # データの取得（日足）
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas

    # 前処理（線形トレンド除去）
    detrended_prices = detrend_prices(data)

    # ----------------    日足    ----------------
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("上位10個の周期 (日足):", dominant_periods)

    # サイクルの検出
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
    ) = detect_cycles(detrended_prices)
    print(
        f"ピークサイクル(日足): [avg:{avg_peak_cycle} , med: {median_peak_cycle} , mae: {mean_absolute_error_peak}%"
    )
    print(
        f"谷サイクル(日足): [avg:{avg_trough_cycle} , med: {median_trough_cycle} , mae: {mean_absolute_error_trough}%"
    )

    # ----------------    週足    ----------------
    # データのリサンプリング関数（週足）
    weekly_data = detrended_prices.resample(
        "W"
    ).ffill()  # 週足にリサンプリングし、前週の値を埋める（pandas を必要とする）

    # FFTによるサイクル解析（週足）
    fft_period, fft_amplitude, dominant_periods = fft_analysis(weekly_data.values)
    print("上位10個の周期 (週足):", dominant_periods)

    # サイクルの検出（週足）
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
    ) = detect_cycles(weekly_data)
    print(
        f"ピークサイクル(週足): [avg:{avg_peak_cycle} , med: {median_peak_cycle} , mae: {mean_absolute_error_peak}%"
    )
    print(
        f"谷サイクル(週足): [avg:{avg_trough_cycle} , med: {median_trough_cycle} , mae: {mean_absolute_error_trough}%"
    )

    # ----------------    月足    ----------------

    # データのリサンプリング関数（月足）
    monthly_data = detrended_prices.resample(
        "ME"
    ).ffill()  # 月足にリサンプリングし、前月の値を埋める（pandas を必要とする）

    # FFTによるサイクル解析（月足）
    fft_period, fft_amplitude, dominant_periods = fft_analysis(monthly_data.values)
    print("上位10個の周期 (月足):", dominant_periods)

    # サイクルの検出（月足）
    (
        peaks,
        troughs,
        avg_peak_cycle,
        median_peak_cycle,
        mean_absolute_error_peak,
        avg_trough_cycle,
        median_trough_cycle,
        mean_absolute_error_trough,
    ) = detect_cycles(monthly_data)
    print(
        f"ピークサイクル(月足): [avg:{avg_peak_cycle} , med: {median_peak_cycle} , mae: {mean_absolute_error_peak}%"
    )
    print(
        f"谷サイクル(月足): [avg:{avg_trough_cycle} , med: {median_trough_cycle} , mae: {mean_absolute_error_trough}%"
    )


if __name__ == "__main__":
    main()
