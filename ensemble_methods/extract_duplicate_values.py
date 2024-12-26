import pandas as pd


def extract_duplicate_values(selected_signals, min_overlap_count=2):
    print(f"len(selected_signals) : {len(selected_signals)}")

    duplicated_values = {}
    symbols = list(selected_signals.values())[0].keys()  # 全てのシンボルを取得
    print(f"symbols: {symbols}")
    for symbol in symbols:
        # 各モデルが予測した日付リストをすべて結合し、重複を確認
        all_dates = [
            date
            for model_signals in selected_signals.values()
            for date in model_signals[symbol]
        ]
        date_counts = pd.Series(all_dates).value_counts()

        # print(f"date_counts: \n{date_counts}")
        # 重複回数が指定された回数以上のものだけを残す
        duplicate_dates = date_counts[date_counts >= min_overlap_count].index.tolist()

        if duplicate_dates:
            duplicated_values[symbol] = sorted(duplicate_dates)  # 日付をソートして保存

    #     print(f"len(duplicated_values[symbol]) : {len(duplicated_values[symbol])}")

    # print(f"len(duplicated_values) : {len(duplicated_values)}")
    # print(f"len(duplicated_values.values()) : {len(duplicated_values.values())}")
    return duplicated_values
