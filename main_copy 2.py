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
    # detrended_first_half = detrend_prices(first_half, remove_trend)
    detrended_second_half = detrend_prices(second_half, remove_trend)
    # detrended_first_quarter = detrend_prices(first_quarter, remove_trend)
    # detrended_second_quarter = detrend_prices(second_quarter, remove_trend)
    # detrended_third_quarter = detrend_prices(third_quarter, remove_trend)
    detrended_fourth_quarter = detrend_prices(fourth_quarter, remove_trend)
    print(f"symbol: {symbol} , start_date: {start_date} , end_date: {end_date}")

    # ----------------    日足    ----------------
    print("----------------    日足    ----------------")
    # データの行数を表示 フーリエ解析の最大周期と一致することを確認した
    # print(f"データ（日）の数: 1/1:{len(detrended_prices)}, 1/2:{len(detrended_first_half)}, 2/2{len(detrended_second_half)}, 1/4:{len(detrended_first_quarter)}, 2/4:{len(detrended_second_quarter)}, 3/4{len(detrended_third_quarter)}, 4/4:{len(detrended_fourth_quarter)}")
    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("fft_analiysis (1/1) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_half.values
    # )
    # print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_quarter.values
    # )
    # print("fft_analiysis (1/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_second_quarter.values
    # )
    # print("fft_analiysis (2/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_third_quarter.values
    # )
    # print("fft_analiysis (3/4) :", dominant_periods)
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
    # print(f"peaks  : {peaks}")
    # print(
    #     f"peak_cycles : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    # )
    print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )
    
    # ----------------    週足    ----------------

    # データのリサンプリング関数（週足）
    detrended_weekly_prices = detrended_prices.resample("W").ffill()
    # detrended_first_half_weekly = detrended_first_half.resample("W").ffill()
    detrended_second_half_weekly = detrended_second_half.resample("W").ffill()
    # detrended_first_quarter_weekly = detrended_first_quarter.resample("W").ffill()
    # detrended_second_quarter_weekly = detrended_second_quarter.resample("W").ffill()
    # detrended_third_quarter_weekly = detrended_third_quarter.resample("W").ffill()
    detrended_fourth_quarter_weekly = detrended_fourth_quarter.resample("W").ffill()
    
    print("----------------    週足    ----------------")
    # データの行数を表示
    # print(f"データ（週）の数: 1/1:{len(detrended_weekly_prices)}, 1/2:{len(detrended_first_half_weekly)}, 2/2:{len(detrended_second_half_weekly)}, 1/4:{len(detrended_first_quarter_weekly)}, 2/4:{len(detrended_second_quarter_weekly)}, 3/4:{len(detrended_third_quarter_weekly)}, 4/4:{len(detrended_fourth_quarter_weekly)}")

    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_weekly_prices.values
    )
    print("fft_analiysis (1/1) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_half_weekly.values
    # )
    # print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half_weekly.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_quarter_weekly.values
    # )
    # print("fft_analiysis (1/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_second_quarter_weekly.values
    # )
    # print("fft_analiysis (2/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_third_quarter_weekly.values
    # )
    # print("fft_analiysis (3/4) :", dominant_periods)
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
    # print(f"peaks  : {peaks}")
    # print(
    #     f"peak_cycles : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    # )
    print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )

    # ----------------    月足    ----------------

    # データのリサンプリング関数（月足）
    detrended_monthly_prices = detrended_prices.resample("ME").ffill()
    # detrended_first_half_monthly = detrended_first_half.resample("ME").ffill()
    detrended_second_half_monthly = detrended_second_half.resample("ME").ffill()
    # detrended_first_quarter_monthly = detrended_first_quarter.resample("ME").ffill()
    # detrended_second_quarter_monthly = detrended_second_quarter.resample("ME").ffill()
    # detrended_third_quarter_monthly = detrended_third_quarter.resample("ME").ffill()
    detrended_fourth_quarter_monthly = detrended_fourth_quarter.resample("ME").ffill()

    print("----------------    月足    ----------------")
    # データの行数を表示
    # print(f"データ（月）の数: 1/1:{len(detrended_monthly_prices)}, 1/2:{len(detrended_first_half_monthly)}, 2/2:{len(detrended_second_half_monthly)}, 1/4:{len(detrended_first_quarter_monthly)}, 2/4:{len(detrended_second_quarter_monthly)}, 3/4:{len(detrended_third_quarter_monthly)}, 4/4:{len(detrended_fourth_quarter_monthly)}")


    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_monthly_prices.values
    )
    print("fft_analiysis (1/1) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_half_monthly.values
    # )
    # print("fft_analiysis (1/2) :", dominant_periods)
    fft_period, fft_amplitude, dominant_periods = fft_analysis(
        detrended_second_half_monthly.values
    )
    print("fft_analiysis (2/2) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_first_quarter_monthly.values
    # )
    # print("fft_analiysis (1/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_second_quarter_monthly.values
    # )
    # print("fft_analiysis (2/4) :", dominant_periods)
    # fft_period, fft_amplitude, dominant_periods = fft_analysis(
    #     detrended_third_quarter_monthly.values
    # )
    # print("fft_analiysis (3/4) :", dominant_periods)
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
    # print(f"peaks  : {peaks}")
    # print(
    #     f"peak_cycles : [avg: {avg_peak_cycle} , med: {median_peak_cycle} , mode: {mode_peak_cycle} , mae: {mean_absolute_error_peak} %]"
    # )
    print(f"troughs: {troughs}")
    print(
        f"trough_cycles: [avg: {avg_trough_cycle} , med: {median_trough_cycle} , mode: {mode_trough_cycle} , mae: {mean_absolute_error_trough} %]"
    )
    


if __name__ == "__main__":
    main()
