from sklearn.impute import SimpleImputer
from data_processing.detrend_prices import detrend_prices
from features.cycle_theory.process_cycle_features import process_cycle_features
from features.technical_indicators.process_technical_features import (
    process_technical_features,
)
from sklearn.decomposition import PCA
import pandas as pd


def create_features_for_dates(
    dates_for_labels,
    daily_datas,
    weekly_datas,
    monthly_datas,
    remove_trend,
    n_components=30,
):
    features = []
    feature_dates = []
    detrended_daily_prices = detrend_prices(daily_datas, remove_trend)
    detrended_weekly_prices = detrend_prices(weekly_datas, remove_trend)
    detrended_monthly_prices = detrend_prices(monthly_datas, remove_trend)

    for date in dates_for_labels:
        recent_detrended_prices = {
            "d": detrended_daily_prices.loc[:date].tail(90),
            "w": detrended_weekly_prices.loc[:date].tail(60),
            "m": detrended_monthly_prices.loc[:date].tail(36),
        }

        feature = {}
        for freq, prices in recent_detrended_prices.items():
            prefix = freq
            feature.update(process_cycle_features(prices, prefix))

        # 最近のデータを取得 価格・出来高は過去10個分さかのぼる、出来高はそこから5個分の移動平均を算出
        recent_datas = {
            "d": daily_datas.loc[:date].tail(20),
            "w": weekly_datas.loc[:date].tail(20),
        }

        # 出来高特徴量の計算
        for freq, datas in recent_datas.items():
            prefix = freq
            feature.update(process_technical_features(datas, prefix))

        features.append(feature)
        feature_dates.append(date)

    # 特徴量をデータフレームに変換
    features_df = pd.DataFrame(features)

    # PCA前の特徴量の数を表示
    print(f"PCA前の特徴量の数: {features_df.shape[1]}")

    # 欠損値を補完 (平均値で補完)
    imputer = SimpleImputer(strategy="mean")
    features_df = pd.DataFrame(
        imputer.fit_transform(features_df), columns=features_df.columns
    )

    # PCAの適用
    if not features_df.empty:
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features_df)

        # PCA後の特徴量の数を表示
        print(f"PCA後の特徴量の数: {reduced_features.shape[1]}")

        # 元のデータ形式に戻す
        reduced_features = pd.DataFrame(
            reduced_features, index=features_df.index
        ).to_dict(orient="records")
    else:
        reduced_features = []

    return reduced_features, feature_dates
