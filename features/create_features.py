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
    detrended_daily_prices = detrend_prices(daily_data, remove_trend)
    detrended_weekly_prices = detrend_prices(weekly_data, remove_trend)
    detrended_monthly_prices = detrend_prices(monthly_data, remove_trend)

    # ラベルデータを取引開始日以降に限定
    labeled_data = labeled_data[labeled_data.index >= trade_start_date]

    # 正解ラベルと不正解ラベルのデータを取得
    correct_label_dates = labeled_data[labeled_data["Label"] == 1].index
    all_incorrect_label_dates = labeled_data[labeled_data["Label"] == 0].index

    # サンプリングされた不正解ラベルの数を計算
    num_correct_labels = len(correct_label_dates)
    num_incorrect_labels = num_correct_labels * data_numbers

    # 不正解ラベルをサンプリング
    sampled_incorrect_label_dates = (
        labeled_data[labeled_data["Label"] == 0]
        .sample(num_incorrect_labels, random_state=42)
        .index
    )

    # ==========================
    # 特徴量の作成（全てのデータ）
    # ==========================
    # 正解ラベルの特徴量を作成
    features_correct_all, dates_correct_all = create_features_for_dates(
        correct_label_dates,
        detrended_daily_prices,
        detrended_weekly_prices,
        detrended_monthly_prices,
    )

    # 不正解ラベルの特徴量を作成（すべての不正解ラベル）
    features_incorrect_all, dates_incorrect_all = create_features_for_dates(
        all_incorrect_label_dates,
        detrended_daily_prices,
        detrended_weekly_prices,
        detrended_monthly_prices,
    )

    # 特徴量と日付を結合（全データ）
    features_all = features_correct_all + features_incorrect_all
    feature_dates_all = dates_correct_all + dates_incorrect_all

    # データフレーム化（全データ）
    features_all_df = pd.DataFrame(features_all, index=feature_dates_all)

    # ラベルを追加（全データ）
    features_all_df["Label"] = labeled_data["Label"].loc[feature_dates_all]

    # ==================================
    # 特徴量の作成（サンプリングデータ）
    # ==================================
    # 不正解ラベルの特徴量を作成（サンプリングされた不正解ラベル）
    features_incorrect_sampled, dates_incorrect_sampled = create_features_for_dates(
        sampled_incorrect_label_dates,
        detrended_daily_prices,
        detrended_weekly_prices,
        detrended_monthly_prices,
    )

    # 特徴量と日付を結合（サンプリングデータ）
    features_sampled = features_correct_all + features_incorrect_sampled
    feature_dates_sampled = dates_correct_all + dates_incorrect_sampled

    # データフレーム化（サンプリングデータ）
    features_df = pd.DataFrame(features_sampled, index=feature_dates_sampled)

    # ラベルを追加（サンプリングデータ）
    features_df["Label"] = labeled_data["Label"].loc[feature_dates_sampled]

    return features_df, features_all_df


def create_features_for_dates(dates, daily_prices, weekly_prices, monthly_prices):
    features = []
    feature_dates = []
    for date in dates:
        recent_prices = {
            "daily": daily_prices.loc[:date].tail(90),
            "weekly": weekly_prices.loc[:date].tail(60),
            "monthly": monthly_prices.loc[:date].tail(36),
        }
        feature = create_individual_features(recent_prices)
        features.append(feature)
        feature_dates.append(date)
    return features, feature_dates


def create_individual_features(recent_prices):
    feature = {}
    for freq, prices in recent_prices.items():
        prefix = freq
        feature.update(process_frequency_features(prices, prefix))
    return feature


def process_frequency_features(prices, prefix):
    feature = {}
    if len(prices) > 0:
        # サイクル検出
        (
            _,
            troughs,
            _,
            _,
            _,
            _,
            avg_trough_cycle,
            median_trough_cycle,
            _,
            mode_trough_cycle,
        ) = detect_cycles(prices)
        if len(troughs) > 1:
            feature[f"{prefix}_avg_trough_cycle"] = avg_trough_cycle
            feature[f"{prefix}_med_trough_cycle"] = median_trough_cycle
            feature[f"{prefix}_mode_trough_cycle"] = mode_trough_cycle
        else:
            feature[f"{prefix}_avg_trough_cycle"] = np.nan
            feature[f"{prefix}_med_trough_cycle"] = np.nan
            feature[f"{prefix}_mode_trough_cycle"] = np.nan

        # FFT解析
        _, _, dominant_periods = fft_analysis(prices.values)
        feature[f"{prefix}_fft_dominant_period"] = (
            dominant_periods[0] if len(dominant_periods) > 0 else np.nan
        )
    else:
        feature[f"{prefix}_avg_trough_cycle"] = np.nan
        feature[f"{prefix}_med_trough_cycle"] = np.nan
        feature[f"{prefix}_mode_trough_cycle"] = np.nan
        feature[f"{prefix}_fft_dominant_period"] = np.nan
    return feature
