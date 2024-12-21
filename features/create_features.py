import pandas as pd
import numpy as np
from features.feature_functions import create_features_for_dates

def create_features(daily_data, trade_start_date, labeled_data, data_numbers):
    weekly_data = daily_data.resample("W").ffill()
    monthly_data = daily_data.resample("ME").ffill()

    # ラベルデータを取引開始日以降に限定
    labeled_data = labeled_data[labeled_data.index >= trade_start_date]

    # 正解ラベルと不正解ラベルのデータを取得
    correct_label_dates = labeled_data[labeled_data["Label"] == 1].index
    all_incorrect_label_dates = labeled_data[labeled_data["Label"] == 0].index

    # サンプリングされた不正解ラベルの数を計算
    num_correct_labels = len(correct_label_dates)
    num_incorrect_labels = min(len(all_incorrect_label_dates), num_correct_labels * data_numbers)

    # 不正解ラベルをサンプリング
    if num_incorrect_labels > 0:
        sampled_incorrect_label_dates = (
            labeled_data[labeled_data["Label"] == 0]
            .sample(num_incorrect_labels, random_state=42)
            .index
        )
    else:
        sampled_incorrect_label_dates = pd.Index([])

    # ==========================
    # 特徴量の作成（全てのデータ）
    # ==========================
    # 正解ラベルの特徴量を作成
    features_correct_all, dates_correct_all = create_features_for_dates(
        correct_label_dates,
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True
    )

    # 不正解ラベルの特徴量を作成（すべての不正解ラベル）
    features_incorrect_all, dates_incorrect_all = create_features_for_dates(
        all_incorrect_label_dates,
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True
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
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True
    )

    # 特徴量と日付を結合（サンプリングデータ）
    features_sampled = features_correct_all + features_incorrect_sampled
    feature_dates_sampled = dates_correct_all + dates_incorrect_sampled

    # データフレーム化（サンプリングデータ）
    features_df = pd.DataFrame(features_sampled, index=feature_dates_sampled)

    # ラベルを追加（サンプリングデータ）
    features_df["Label"] = labeled_data["Label"].loc[feature_dates_sampled]

    return features_df, features_all_df
