import numpy as np
from features.technical_indicators.volume_analysis import calculate_volume_features


def process_technical_features(prices, prefix, date):
    feature = {}
    if len(prices) > 0:
        volume_features = calculate_volume_features(prices, prefix)
        if date in volume_features.index:
            for col in volume_features.columns:
                new_col = f"{prefix}_{col.split('_', 1)[1]}"
                feature[new_col] = volume_features.at[date, col]
        else:
            for col in volume_features.columns:
                new_col = f"{prefix}_{col.split('_', 1)[1]}"
                feature[new_col] = np.nan
    return feature
