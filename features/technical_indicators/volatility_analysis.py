import pandas as pd


def calculate_volatility_features(prices):
    """
    10個前のデータを基準に、9個前が何パーセント(小数第一位)変化したか
    10個前のデータを基準に、8個前が何パーセント(小数第一位)変化したか
    ...
    10個前のデータを基準に、1個前が何パーセント(小数第一位)変化したか
    10個前のデータを基準に、現在が何パーセント(小数第一位)変化したか

    5個前のデータを基準に、4個前が何パーセント(小数第一位)変化したか
    5個前のデータを基準に、3個前が何パーセント(小数第一位)変化したか
    ...
    5個前のデータを基準に、1個前が何パーセント(小数第一位)変化したか
    5個前のデータを基準に、現在が何パーセント(小数第一位)変化したか

    過去5個の平均と当日の値の差割合を過去10日分

    Args:
        prices (pandas.Series): 価格データを含むシリーズ
    Returns:
        pandas.DataFrame: 10日前基準の推移と5日前基準の推移を含むデータフレーム
    """
    feature = pd.DataFrame(index=prices.index)
    data_length = len(prices)

    # 10個前基準の変化率を計算
    if data_length >= 10:
        base_price_10 = prices.shift(10)
        for i in range(9, -1, -1):
            feature[f"per_chg_fm_10_{i}"] = (
                (prices.shift(i) - base_price_10) / base_price_10 * 100
            ).round(1)

    # 15個前基準の変化率を計算
    if data_length >= 15:
        base_price_15 = prices.shift(15)
        for i in range(14, -1, -1):
            feature[f"per_chg_fm_15_{i}"] = (
                (prices.shift(i) - base_price_15) / base_price_15 * 100
            ).round(1)

    # 追加 過去5個の平均と当日の値の差割合を過去10日分
    if data_length >= 10:
        for i in range(9, -1, -1):
            feature[f"per_5ma_10_{i}"] = (
                (prices.shift(i) - prices.shift(i).rolling(window=5).mean())
                / prices.shift(i).rolling(window=5).mean()
                * 100
            ).round(1)

    return feature
