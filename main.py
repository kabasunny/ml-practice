# main.py

from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from cycle_theory.peak_trough.detect_cycles import detect_cycles
from cycle_theory.fourier.fft_analysis import fft_analysis


# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2004-01-01"
    end_date = "2023-12-31"

    # データの取得（日足）
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas

    # データ分割
    n = len(data)
    # 前半・後半に分割
    first_half = data.iloc[: n // 2]
    second_half = data.iloc[n // 2 :]

    # 4分割に分割
    first_quarter = data.iloc[: n // 4]
    second_quarter = data.iloc[n // 4 : n // 2]
    third_quarter = data.iloc[n // 2 : 3 * n // 4]
    fourth_quarter = data.iloc[3 * n // 4 :]

    # トレンドを除去する場合 True トレンド含む場合 False に設定
    remove_trend = True
    # remove_trend = False

    # 前処理
    if remove_trend:
        print("トレンド除去を行う")
    else:
        print("トレンド除去を行わない")

    detrended_prices = detrend_prices(data, remove_trend)
    detrended_first_half = detrend_prices(first_half, remove_trend)
    detrended_second_half = detrend_prices(second_half, remove_trend)
    detrended_first_quarter = detrend_prices(first_quarter, remove_trend)
    detrended_second_quarter = detrend_prices(second_quarter, remove_trend)
    detrended_third_quarter = detrend_prices(third_quarter, remove_trend)
    detrended_fourth_quarter = detrend_prices(fourth_quarter, remove_trend)

    print(f"symbol: {symbol} , start_date: {start_date} , end_date: {end_date}")

    # ----------------    日足    ----------------
    print("----------------    日足    ----------------")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_half.values
    )
    print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_quarter.values
    )
    print("fft_analiysis (1/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_quarter.values
    )
    print("fft_analiysis (2/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_third_quarter.values
    )
    print("fft_analiysis (3/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_fourth_quarter.values
    )
    print("fft_analiysis (4/4) :", dominant_periods)

    # サイクルの検出
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
        f"peak_cycles     : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles   : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # ----------------    週足    ----------------

    # データのリサンプリング関数（週足）
    detrended_prices = detrended_prices.resample(
        "W"
    ).ffill()  # 週足にリサンプリングし、前週の値を埋める（pandas を必要とする）

    print("----------------    週足    ----------------")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_half.values
    )
    print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_quarter.values
    )
    print("fft_analiysis (1/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_quarter.values
    )
    print("fft_analiysis (2/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_third_quarter.values
    )
    print("fft_analiysis (3/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_fourth_quarter.values
    )
    print("fft_analiysis (4/4) :", dominant_periods)

    # サイクルの検出
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
        f"peak_cycles     : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles   : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # ----------------    月足    ----------------

    # データのリサンプリング関数（月足）
    detrended_prices = detrended_prices.resample(
        "ME"
    ).ffill()  # 月足にリサンプリングし、前月の値を埋める（pandas を必要とする）

    print("----------------    月足    ----------------")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_half.values
    )
    print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_first_quarter.values
    )
    print("fft_analiysis (1/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_quarter.values
    )
    print("fft_analiysis (2/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_third_quarter.values
    )
    print("fft_analiysis (3/4) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_fourth_quarter.values
    )
    print("fft_analiysis (4/4) :", dominant_periods)

    # サイクルの検出
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
        f"peak_cycles     : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    )
    print(
        f"trough_cycles   : [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )


if __name__ == "__main__":
    main()
