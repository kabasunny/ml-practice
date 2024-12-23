import pandas as pd


def calculate_volume_features(data, prefix):
    """
    日足または週足の出来高の当日、1日前、2日前、…9日前の値と、
    それらの日付における過去5日分（または5週分）の出来高移動平均の値を返す関数
    Args:
        data (pandas.DataFrame): 出来高データを含むデータフレーム。'Volume' 列が必要。
        prefix (str): "d"（日足）または "w"（週足）。
    Returns:
        pandas.DataFrame: 出来高の値と出来高移動平均の値を含むデータフレーム
    """
    feature = pd.DataFrame(index=data.index)

    # 出来高の当日、1日前、2日前、…9日前の値を追加
    for i in range(10):
        feature[f"{prefix}_vol_t-{i}"] = data["Volume"].shift(i)

    # 出来高の過去5日分（または5週分）の移動平均を追加
    for i in range(10):
        feature[f"{prefix}_vol_ma_5_t-{i}"] = (
            data["Volume"].shift(i).rolling(window=5).mean()
        )

    return feature
