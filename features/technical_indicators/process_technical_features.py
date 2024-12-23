import numpy as np
from features.technical_indicators.volume_analysis import calculate_volume_features
from features.technical_indicators.volatility_analysis import (
    calculate_volatility_features,
)


def process_technical_features(datas, prefix):
    feature = {}
    if len(datas) > 0:
        # 出来高特徴量を計算
        volume_features = calculate_volume_features(datas["Volume"])
        for col in volume_features.columns:
            vol_col = f"{prefix}_{col}"
            feature[vol_col] = volume_features[col].iloc[-1]  # 最後の日の値を取得

        # 価格変動率特徴量を計算
        volatility_features = calculate_volatility_features(datas["Close"])
        for col in volatility_features.columns:
            volat_col = f"{prefix}_{col}"
            feature[volat_col] = volatility_features[col].iloc[-1]  # 最後の日の値を取得

    return feature
