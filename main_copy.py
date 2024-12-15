# main.py

from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from cycle_theory.peak_trough.detect_cycles import detect_cycles
from cycle_theory.fourier.fft_analysis import fft_analysis
import pandas as pd


# トヨタ自動車 ( 7203
# ソニー ( 6758
# 任天堂 ( 7974
# 日産自動車 ( 7201
# ホンダ ( 7267
# 三菱UFJフィナンシャル・グループ ( 8306
# キヤノン ( 7751
# パナソニック ( 6752
# ソフトバンクグループ ( 9984
# 東京エレクトロン ( 8035


# メイン処理
def main():
    symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
    trade_start_date = pd.Timestamp("2022-02-01")  # 買いを入れる日
    period_days = 365 * 10  # 前後1年を期間とする例

    # start_date と end_date を trade_start_date を基に設定
    start_date = trade_start_date - pd.Timedelta(days=period_days)
    end_date = trade_start_date + pd.Timedelta(days=period_days)

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
        print("トレンド除去を実施")
    else:
        print("トレンド除去を未実施")

    detrended_prices = detrend_prices(data, remove_trend)
    detrended_second_half = detrend_prices(second_half, remove_trend)
    detrended_fourth_quarter = detrend_prices(fourth_quarter, remove_trend)
    print(f"symbol: {symbol} , start_date: {start_date} , end_date: {end_date}")

    # ----------------    日足    ----------------
    print("----------------    日足    ----------------")# FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
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
    # print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )
    




    # ----------------    週足    ----------------

    # データのリサンプリング関数（週足）
    detrended_weekly_prices = detrended_prices.resample("W").ffill()
    detrended_second_half_weekly = detrended_second_half.resample("W").ffill()
    detrended_fourth_quarter_weekly = detrended_fourth_quarter.resample("W").ffill()

    # ラベルの初期化
    weekly_data = detrended_weekly_prices.copy()
    weekly_data['Label'] = 0
    
    print("----------------    週足    ----------------")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_weekly_prices.values
    )
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half_weekly.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_fourth_quarter_weekly.values
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
    ) = detect_cycles(detrended_weekly_prices)
    # print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    
    # # トラフのインデックスに対応する日付を取得
    # trough_dates = weekly_data.iloc[troughs].index
    
    # # トラフの日付のラベルを1に設定
    # weekly_data.loc[trough_dates, 'Label'] = 1





    # ----------------    月足    ----------------

    # データのリサンプリング関数（月足）
    detrended_monthly_prices = detrended_prices.resample("ME").ffill()
    detrended_second_half_monthly = detrended_second_half.resample("ME").ffill()
    detrended_fourth_quarter_monthly = detrended_fourth_quarter.resample("ME").ffill()

    print("----------------    月足    ----------------")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_monthly_prices.values
    )
    print("fft_analiysis (1/1) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half_monthly.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_fourth_quarter_monthly.values
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
    ) = detect_cycles(detrended_monthly_prices)
    # print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )
    


if __name__ == "__main__":
    main()
