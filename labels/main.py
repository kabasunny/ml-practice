import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import pandas as pd
import matplotlib.pyplot as plt
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels


def main():
    # 株価データの取得
    symbol = "7203.T"  # トヨタ自動車の例
    start_date = "2003-01-01"
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
    plt.plot(daily_data.index, daily_data["Close"], label="Daily Data (Close)")
    plt.scatter(
        labeled_data[labeled_data["Label"] == 1].index,
        labeled_data[labeled_data["Label"] == 1]["Close"],
        color="red",
        label="Label",
    )
    plt.title("Daily Data with Labels")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
