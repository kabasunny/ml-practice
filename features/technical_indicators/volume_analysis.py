import pandas as pd


def calculate_volume_features(valumes):
    """
    出来高で、現在、1個前、2個前、…(データ数-1)個前の値と、
    それらのデータにおける過去5個分の移動平均の値を返す関数
    Args:
        valumes (pandas.Series): 出来高データを含むシリーズ
    Returns:
        pandas.DataFrame: 出来高の値と出来高移動平均の値を含むデータフレーム
    """
    feature = pd.DataFrame(index=valumes.index)
    data_length = len(valumes)

    # 出来高の当日、1日前、2日前、…(データ数-1)日前の値を追加
    for i in range(min(10, data_length)):
        feature[f"vol_t-{i}"] = valumes.shift(i)

    
    # いらなそう
    # 出来高の過去5日分（または5週分）の移動平均を追加
    # for i in range(min(10, data_length)):
    #     feature[f"vol_5ma_t-{i}"] = valumes.shift(i).rolling(window=5).mean()

    return feature
