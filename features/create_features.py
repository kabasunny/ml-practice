import pandas as pd
import numpy as np
from cycle_theory.fourier.fft_analysis import fft_analysis
from cycle_theory.peak_trough.detect_cycles import detect_cycles
from data_processing.detrend_prices import detrend_prices

def create_features(daily_data, trade_start_date, labeled_data, data_numbers):
    weekly_data = daily_data.resample("W").ffill()
    monthly_data = daily_data.resample("ME").ffill()

    # トレンド除去
    remove_trend = True
    detrended_prices = detrend_prices(daily_data, remove_trend)
    detrended_weekly_prices = detrend_prices(weekly_data, remove_trend)
    detrended_monthly_prices = detrend_prices(monthly_data, remove_trend)

    # 特徴量の作成
    features = []
    feature_dates = []  # 特徴量作成時の日付を保持するリスト

    # 正解ラベルの特徴量を全数作成
    for date in labeled_data[labeled_data['Label'] == 1].index:
        if date >= trade_start_date:
            recent_daily_prices = detrended_prices.loc[:date].tail(90)
            recent_weekly_prices = detrended_weekly_prices.loc[:date].tail(60)
            recent_monthly_prices = detrended_monthly_prices.loc[:date].tail(36)

            feature = create_individual_features(recent_daily_prices, recent_weekly_prices, recent_monthly_prices)
            features.append(feature)
            feature_dates.append(date)  # 日付をリストに追加

    # 正解ラベル数に応じて不正解ラベルの特徴量をランダムに選択して作成
    num_correct_labels = len(feature_dates)
    num_incorrect_labels = num_correct_labels * data_numbers
    incorrect_label_dates = labeled_data[(labeled_data['Label'] == 0) & (labeled_data.index >= trade_start_date)].sample(num_incorrect_labels).index

    for date in incorrect_label_dates:
        recent_daily_prices = detrended_prices.loc[:date].tail(90)
        recent_weekly_prices = detrended_weekly_prices.loc[:date].tail(60)
        recent_monthly_prices = detrended_monthly_prices.loc[:date].tail(36)

        feature = create_individual_features(recent_daily_prices, recent_weekly_prices, recent_monthly_prices)
        features.append(feature)
        feature_dates.append(date)  # 日付をリストに追加

    # 特徴量のデータフレーム化
    features_df = pd.DataFrame(features, index=feature_dates)

    # ラベルを結合
    features_df['Label'] = labeled_data['Label'].loc[feature_dates]

    return features_df

def create_individual_features(detrended_prices, detrended_weekly_prices, detrended_monthly_prices):
    feature = {}

    # 日足の特徴量
    if len(detrended_prices) > 0:
        _, daily_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_prices)
        if len(daily_troughs) > 1:
            feature['daily_avg_trough_cycle'] = avg_trough_cycle
            feature['daily_med_trough_cycle'] = median_trough_cycle
            feature['daily_mode_trough_cycle'] = mode_trough_cycle
        else:
            feature['daily_avg_trough_cycle'] = np.nan
            feature['daily_med_trough_cycle'] = np.nan
            feature['daily_mode_trough_cycle'] = np.nan

        _, _, dominant_periods = fft_analysis(detrended_prices.values)
        feature['daily_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan
    else:
        feature['daily_avg_trough_cycle'] = np.nan
        feature['daily_med_trough_cycle'] = np.nan
        feature['daily_mode_trough_cycle'] = np.nan
        feature['daily_fft_dominant_period'] = np.nan

    # 週足の特徴量
    if len(detrended_weekly_prices) > 0:
        _, weekly_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_weekly_prices)
        if len(weekly_troughs) > 1:
            feature['weekly_avg_trough_cycle'] = avg_trough_cycle
            feature['weekly_med_trough_cycle'] = median_trough_cycle
            feature['weekly_mode_trough_cycle'] = mode_trough_cycle
        else:
            feature['weekly_avg_trough_cycle'] = np.nan
            feature['weekly_med_trough_cycle'] = np.nan
            feature['weekly_mode_trough_cycle'] = np.nan

        _, _, dominant_periods = fft_analysis(detrended_weekly_prices.values)
        feature['weekly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan
    else:
        feature['weekly_avg_trough_cycle'] = np.nan
        feature['weekly_med_trough_cycle'] = np.nan
        feature['weekly_mode_trough_cycle'] = np.nan
        feature['weekly_fft_dominant_period'] = np.nan

    # 月足の特徴量
    if len(detrended_monthly_prices) > 0:
        _, monthly_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_monthly_prices)
        if len(monthly_troughs) > 1:
            feature['monthly_avg_trough_cycle'] = avg_trough_cycle
            feature['monthly_med_trough_cycle'] = median_trough_cycle
            feature['monthly_mode_trough_cycle'] = mode_trough_cycle
        else:
            feature['monthly_avg_trough_cycle'] = np.nan
            feature['monthly_med_trough_cycle'] = np.nan
            feature['monthly_mode_trough_cycle'] = np.nan

        _, _, dominant_periods = fft_analysis(detrended_monthly_prices.values)
        feature['monthly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan
    else:
        feature['monthly_avg_trough_cycle'] = np.nan
        feature['monthly_med_trough_cycle'] = np.nan
        feature['monthly_mode_trough_cycle'] = np.nan
        feature['monthly_fft_dominant_period'] = np.nan

    return feature
