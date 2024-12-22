import numpy as np
from data_processing.detrend_prices import detrend_prices
from features.process_frequency_features import process_frequency_features
from features.technical_indicators.volume_analysis import calculate_volume_features


def create_features_for_dates(
    dates_for_labels, daily_prices, weekly_prices, monthly_prices, remove_trend
):
    features = []
    feature_dates = []

    for date in dates_for_labels:  # ここを修正
        # daily_prices を処理して detrended_daily_prices, detrended_weekly_prices, detrended_monthly_prices を作成
        detrended_daily_prices = detrend_prices(daily_prices, remove_trend)
        detrended_weekly_prices = detrend_prices(weekly_prices, remove_trend)
        detrended_monthly_prices = detrend_prices(monthly_prices, remove_trend)

        # 最近の価格データを取得
        recent_prices = {
            "d": detrended_daily_prices.loc[:date].tail(90),
            "w": detrended_weekly_prices.loc[:date].tail(60),
            "m": detrended_monthly_prices.loc[:date].tail(36),
        }

        feature = {}
        for freq, prices in recent_prices.items():
            prefix = freq
            feature.update(process_frequency_features(prices, prefix))

        # 出来高特徴量の計算
        for freq, prices in [("d", daily_prices), ("w", weekly_prices)]:
            volume_features = calculate_volume_features(
                prices.loc[:date], frequency=freq
            )
            if date in volume_features.index:
                for col in volume_features.columns:
                    new_col = f"{freq}_{col.split('_', 1)[1]}"
                    feature[new_col] = volume_features.at[date, col]
            else:
                for col in volume_features.columns:
                    new_col = f"{freq}_{col.split('_', 1)[1]}"
                    feature[new_col] = np.nan

        features.append(feature)
        feature_dates.append(date)

    return features, feature_dates
