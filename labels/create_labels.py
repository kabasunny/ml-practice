import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
from detect_troughs import detect_troughs
from data_processing.fetch_stock_data import fetch_stock_data
import matplotlib.pyplot as plt

def create_labels(daily_data):
    """
    日足の終値が週足トラフと同じ値であるものをラベル付けする関数
    """

    # 週足のピーク・トラフ解析
    troughs, _, _, _, _ = detect_troughs(daily_data['Close'])

    # daily_data が Series の場合は DataFrame に変換
    if isinstance(daily_data, pd.Series):
        daily_data = daily_data.to_frame(name="Close")
        print('to pd.DataFrame daily_data = daily_data.to_frame(name="Close")')

    # Label 列を初期化
    daily_data['Label'] = 0

    selected_troughs = []

    # 検出幅の設定
    pre_x = 7  # 前側の検出幅
    post_x = 33  # 後ろ側の検出幅

    # 最初のトラフデータ点から前後の検出幅を持たせて、次の連続データを選び最小値を検出
    i = 0
    while i < len(troughs):
        # 前後の検出幅を持たせて、次の連続データを選び最小値を検出
        start_idx = max(troughs[i] - pre_x, 0)
        end_idx = min(troughs[i] + post_x + 1, len(daily_data))
        sampling_window = daily_data.iloc[start_idx:end_idx]
        min_close_value = sampling_window['Close'].min()  # 最小値を検出
        min_close_date = sampling_window[sampling_window['Close'] == min_close_value].index[0]  # index[0] を使用することで、最初に一致する行の日付を取得

        # 条件に応じてトラフを追加または削除
        if selected_troughs:
            last_trough = selected_troughs[-1]
            if (min_close_date - last_trough).days > (pre_x + post_x):
                selected_troughs.append(min_close_date)
            elif (min_close_date - last_trough).days <= (pre_x + post_x) and min_close_value < daily_data.loc[last_trough, 'Close']:
                selected_troughs[-1] = min_close_date
        else:
            selected_troughs.append(min_close_date)

        i += 1

    # ラベルを付ける
    for trough_date in selected_troughs:
        daily_data.at[trough_date, 'Label'] = 1

    # ラベル間のスパン（データ個数）を計算してprint
    # for i in range(1, len(selected_troughs)):
    #     span = (selected_troughs[i] - selected_troughs[i - 1]).days
    #     print(f'Span between index {i-1} and index {i}: {span} days')

    return daily_data

def main():
    # 株価データの取得
    symbol = "7203.T"  # トヨタ自動車の例
    start_date = "2014-01-01"
    end_date = "2023-12-31"
    
    daily_data = fetch_stock_data(symbol, start_date, end_date)

    # ラベルの作成とトラフの取得
    labeled_data = create_labels(daily_data)

    # # データの表示
    # print("\nトラフの値:")
    # trough_dates = labeled_data[labeled_data['Label'] == 1].index
    # trough_values = labeled_data[labeled_data['Label'] == 1]['Close']
    # for date, value in zip(trough_dates, trough_values):
    #     print(f"Date: {date}, Close: {value}")

    # print("\n正解のラベルのみ表示:")
    # print(labeled_data[labeled_data['Label'] == 1])

    # # 配列の長さを表示
    # print("\n配列の長さ:")
    # print(f"trough_dates length: {len(trough_dates)}")
    # print(f"labeled_data length: {len(labeled_data[labeled_data['Label'] == 1])}")

    # チャートと正解ラベルのプロット
    plt.figure(figsize=(14, 7))
    plt.plot(daily_data.index, daily_data['Close'], label='Daily Data (Close)')
    plt.scatter(labeled_data[labeled_data['Label'] == 1].index, labeled_data[labeled_data['Label'] == 1]['Close'], color='red', label='Label')
    plt.title('Daily Data with Labels')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
