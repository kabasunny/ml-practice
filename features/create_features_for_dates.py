from data_processing.detrend_prices import detrend_prices
from features.cycle_theory.process_cycle_features import process_cycle_features
from features.technical_indicators.process_technical_features import (
    process_technical_features,
)


def create_features_for_dates(
    dates_for_labels, daily_datas, weekly_datas, monthly_datas, remove_trend
):
    features = []
    feature_dates = []
    detrended_daily_prices = detrend_prices(daily_datas, remove_trend)
    detrended_weekly_prices = detrend_prices(weekly_datas, remove_trend)
    detrended_monthly_prices = detrend_prices(monthly_datas, remove_trend)

    for date in dates_for_labels:

        # 最近の価格データを取得
        recent_detrended_prices = {
            "d": detrended_daily_prices.loc[:date].tail(90),
            "w": detrended_weekly_prices.loc[:date].tail(60),
            "m": detrended_monthly_prices.loc[:date].tail(36),
        }

        feature = {}
        for freq, prides in recent_detrended_prices.items():
            prefix = freq
            feature.update(process_cycle_features(prides, prefix))

        # 最近のデータを取得  価格・出来高は過去10個分さかのぼる、出来高はそこから5個分の移動平均を算出
        recent_datas = {
            "d": daily_datas.loc[:date].tail(15),
            "w": weekly_datas.loc[:date].tail(15),
        }

        # 出来高特徴量の計算
        for freq, datas in recent_datas.items():
            prefix = freq
            feature.update(process_technical_features(datas, prefix))

        features.append(feature)
        feature_dates.append(date)

    return features, feature_dates
