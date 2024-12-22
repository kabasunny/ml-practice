from labels.detect_troughs import detect_troughs


# pandas.DataFrame を取り pandas.DataFrame を戻す
def create_labels(daily_data):
    """
    日足の終値が週足トラフと同じ値であるものをラベル付けする関数
    """
    # print(f"daily_data:\n{daily_data}")  # pandas.DataFrame
    # print(f'daily_data["Close"]:\n{daily_data["Close"]}')  # pandas.Series

    # 日足トラフのみ解析
    troughs, _, _, _, _ = detect_troughs(daily_data["Close"])

    # Label 列を初期化
    daily_data["Label"] = (
        0  # 元々存在しなかった Label 列が daily_data データフレームに追加される
    )
    selected_troughs = []

    # 日足トラフの検出幅の設定
    pre_x = 7  # 前側の検出幅
    post_x = 33  # 後ろ側の検出幅

    # 最初のトラフデータ点から前後の検出幅を持たせて、次の連続データを選び最小値を検出
    i = 0
    while i < len(troughs):
        # 前後の検出幅を持たせて、次の連続データを選び最小値を検出
        start_idx = max(troughs[i] - pre_x, 0)
        end_idx = min(troughs[i] + post_x + 1, len(daily_data))
        sampling_window = daily_data.iloc[start_idx:end_idx]
        min_close_value = sampling_window["Close"].min()  # 最小値を検出
        min_close_date = sampling_window[
            sampling_window["Close"] == min_close_value
        ].index[
            0
        ]  # index[0] を使用することで、最初に一致する行の日付を取得

        # 条件に応じてトラフを追加または削除
        if selected_troughs:
            last_trough = selected_troughs[-1]
            # daily_data のインデックスは DatetimeIndex 型で、各要素が pandas.Timestamp 型
            # pandas.Timestamp 型の日付同士の差分を計算すると timedelta オブジェクトが生成
            # .days 属性を使って日数部分を抽出する
            if (min_close_date - last_trough).days > (pre_x + post_x):
                selected_troughs.append(min_close_date)
            elif (min_close_date - last_trough).days <= (
                pre_x + post_x
            ) and min_close_value < daily_data.loc[last_trough, "Close"]:
                selected_troughs[-1] = min_close_date
        else:
            selected_troughs.append(min_close_date)

        i += 1

    # ラベルを付ける
    for trough_date in selected_troughs:
        daily_data.at[trough_date, "Label"] = 1

    # labeled_data を daily_data として設定
    labeled_data = daily_data

    # print(f"return labeled_data:\n{labeled_data}")  # pandas.DataFrame

    return labeled_data  # 修正: labeled_data を戻す


# [*********************100%***********************]  1 of 1 completed
# daily_data:
#               Open    High     Low   Close    Adj Close    Volume
# Date
# 2003-01-01   638.0   638.0   638.0   638.0   378.700287         0
# 2003-01-02   638.0   638.0   638.0   638.0   378.700287         0
# 2003-01-03   638.0   638.0   638.0   638.0   378.700287         0
# 2003-01-06   650.0   654.0   648.0   650.0   385.823090  17657000
# 2003-01-07   658.0   660.0   644.0   648.0   384.635986  29539000
# ...            ...     ...     ...     ...          ...       ...
# 2023-12-25  2525.0  2553.0  2514.5  2537.0  2470.584473  19273800
# 2023-12-26  2543.0  2544.5  2520.5  2541.0  2474.479980  17223900
# 2023-12-27  2557.5  2583.5  2547.0  2583.0  2515.380371  26896000
# 2023-12-28  2555.0  2573.0  2539.0  2556.0  2489.087158  17822300
# 2023-12-29  2572.0  2615.5  2569.0  2590.5  2522.683838  26860500

# [5222 rows x 6 columns]
# daily_data["Close"]:
# Date
# 2003-01-01     638.0
# 2003-01-02     638.0
# 2003-01-03     638.0
# 2003-01-06     650.0
# 2003-01-07     648.0
#                ...
# 2023-12-25    2537.0
# 2023-12-26    2541.0
# 2023-12-27    2583.0
# 2023-12-28    2556.0
# 2023-12-29    2590.5
# Name: Close, Length: 5222, dtype: float64
# return labeled_data:
#               Open    High     Low   Close    Adj Close    Volume  Label
# Date
# 2003-01-01   638.0   638.0   638.0   638.0   378.700287         0      0
# 2003-01-02   638.0   638.0   638.0   638.0   378.700287         0      0
# 2003-01-03   638.0   638.0   638.0   638.0   378.700287         0      0
# 2003-01-06   650.0   654.0   648.0   650.0   385.823090  17657000      0
# 2003-01-07   658.0   660.0   644.0   648.0   384.635986  29539000      0
# ...            ...     ...     ...     ...          ...       ...    ...
# 2023-12-25  2525.0  2553.0  2514.5  2537.0  2470.584473  19273800      0
# 2023-12-26  2543.0  2544.5  2520.5  2541.0  2474.479980  17223900      0
# 2023-12-27  2557.5  2583.5  2547.0  2583.0  2515.380371  26896000      0
# 2023-12-28  2555.0  2573.0  2539.0  2556.0  2489.087158  17822300      0
# 2023-12-29  2572.0  2615.5  2569.0  2590.5  2522.683838  26860500      0

# [5222 rows x 7 columns]
