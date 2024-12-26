def convert_to_binary_predictions(duplicated_values, test_indices):
    """
    重複する日付をバイナリ予測に変換する関数
    :param duplicated_values: 重複する日付の辞書
    :param test_indices: テストデータのインデックス
    :return: バイナリ予測
    """
    y_pred_binary = [0] * len(test_indices)

    all_duplicated_dates = set()

    # 重複した日付を一つのセットに集約
    for dates in duplicated_values.values():
        all_duplicated_dates.update(dates)

    # 重複日付をバイナリ予測に反映
    for date in all_duplicated_dates:
        if date in test_indices:
            y_pred_binary[test_indices.get_loc(date)] = 1

    return y_pred_binary
