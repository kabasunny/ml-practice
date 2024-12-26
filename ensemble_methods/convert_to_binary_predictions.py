import pandas as pd


def convert_to_binary_predictions(duplicated_values, test_indices, symbols):
    """
    重複する日付をバイナリ予測に変換する関数
    :param duplicated_values: 重複する日付の辞書
    :param test_indices: テストデータのインデックス
    :param symbols: シンボルのリスト
    :return: バイナリ予測
    """
    # y_pred_binary を test_indices と同じ長さで初期化し、すべての要素を 0 に設定
    y_pred_binary = pd.Series([0] * len(test_indices), index=test_indices)

    # 各シンボルについて処理を行う
    for symbol in symbols:
        # シンボルが duplicated_values に存在する場合
        if symbol in duplicated_values:
            # シンボルに対応する重複日付ごとに処理を行う
            for date in duplicated_values[symbol]:
                # 重複日付が test_indices に存在する場合
                if date in test_indices:
                    # 重複日付のインデックスを 1 に設定
                    y_pred_binary.loc[date] = 1

    return y_pred_binary
