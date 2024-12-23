import numpy as np
from features.technical_indicators.volume_analysis import calculate_volume_features


def process_technical_features(datas, prefix):
    feature = {}
    if len(datas) > 0:
        # 出来高特徴量を計算
        volume_features = calculate_volume_features(datas["Volume"])

        # 特定の日付を考慮せず、全ての特徴量を追加
        for col in volume_features.columns:
            new_col = f"{prefix}_{col}"
            feature[new_col] = volume_features[col].iloc[-1]  # 最後の日の値を取得

    return feature
