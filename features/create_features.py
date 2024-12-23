import pandas as pd
from features.create_features_for_dates import create_features_for_dates
from sklearn.model_selection import train_test_split


def create_features(daily_data, trade_start_date, labeled_data, data_numbers):
    weekly_data = daily_data.resample("W").ffill()
    monthly_data = daily_data.resample("ME").ffill()

    # ラベルデータを取引開始日以降に限定
    labeled_data = labeled_data[labeled_data.index >= trade_start_date]

    # 正解ラベルと不正解ラベルのデータを取得
    correct_label_dates = labeled_data[labeled_data["Label"] == 1].index
    all_incorrect_label_dates = labeled_data[labeled_data["Label"] == 0].index

    # 正解ラベルをモデルトレーニング用とモデル評価用にランダムに半分ずつ分ける
    correct_label_dates_for_tr, correct_label_dates_for_ev = train_test_split(
        correct_label_dates, test_size=0.5, random_state=42
    )

    # サンプリングされた不正解ラベルの数を計算
    num_correct_labels = len(correct_label_dates_for_tr)
    num_incorrect_labels = min(
        len(all_incorrect_label_dates), num_correct_labels * data_numbers
    )

    # 不正解ラベルをサンプリング
    if num_incorrect_labels > 0:
        sampled_incorrect_label_dates = (
            labeled_data[labeled_data["Label"] == 0]
            .sample(num_incorrect_labels, random_state=42)
            .index
        )
    else:
        sampled_incorrect_label_dates = pd.Index(
            []
        )  # 空のインデックス pd.Index([]) を設定

    # 特徴量の作成（サンプリングデータ）
    features_correct_tr, dates_correct_tr = create_features_for_dates(
        correct_label_dates_for_tr,
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True,
    )

    features_incorrect_sampled_tr, dates_incorrect_sampled_tr = (
        create_features_for_dates(
            sampled_incorrect_label_dates,
            daily_data,
            weekly_data,
            monthly_data,
            remove_trend=True,
        )
    )

    # 特徴量と日付を結合（トレーニングデータ）
    features_sampled_for_tr = features_correct_tr + features_incorrect_sampled_tr
    feature_dates_sampled_for_tr = dates_correct_tr + dates_incorrect_sampled_tr

    # データフレーム化（トレーニングデータ）
    features_df_for_tr = pd.DataFrame(
        features_sampled_for_tr, index=feature_dates_sampled_for_tr
    )

    # ラベルを追加（トレーニングデータ）
    features_df_for_tr["Label"] = labeled_data["Label"].loc[
        feature_dates_sampled_for_tr
    ]

    # 特徴量の作成（評価データ）
    features_correct_ev, dates_correct_ev = create_features_for_dates(
        correct_label_dates_for_ev,
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True,
    )

    features_incorrect_all, dates_incorrect_all = create_features_for_dates(
        all_incorrect_label_dates,
        daily_data,
        weekly_data,
        monthly_data,
        remove_trend=True,
    )

    # 特徴量と日付を結合（評価用データ）
    features_ev = features_correct_ev + features_incorrect_all
    feature_dates_ev = dates_correct_ev + dates_incorrect_all

    # データフレーム化（評価用データ）
    features_df_for_ev = pd.DataFrame(features_ev, index=feature_dates_ev)

    # トレーニングデータの日付を評価データから除外
    features_df_for_ev = features_df_for_ev.drop(
        features_df_for_tr.index, errors="ignore"
    )

    # ラベルを追加（評価用データ）
    features_df_for_ev["Label"] = labeled_data["Label"].loc[feature_dates_ev]

    # # 不正解ラベルのデータ数と正解ラベルのデータ数を表示
    # label_counts_tr = features_df_for_tr["Label"].value_counts()
    # print(f"tr1不正解ラベル数: {label_counts_tr[0]}")
    # print(f"tr1正解ラベル数  : {label_counts_tr[1]}")

    # # 不正解ラベルのデータ数と正解ラベルのデータ数を表示
    # label_counts_ev = features_df_for_ev["Label"].value_counts()
    # print(f"ev1不正解ラベル数: {label_counts_ev[0]}")
    # print(f"ev1正解ラベル数  : {label_counts_ev[1]}")

    return features_df_for_tr, features_df_for_ev
