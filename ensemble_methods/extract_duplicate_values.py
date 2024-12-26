import pandas as pd


def extract_duplicate_values(selected_signals, min_overlap_count=2):
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

        # 重複回数が指定された回数以上のものだけを残す
        duplicate_dates = date_counts[date_counts >= min_overlap_count].index.tolist()

        if duplicate_dates:
            duplicated_values[symbol] = sorted(duplicate_dates)  # 日付をソートして保存

    return duplicated_values
