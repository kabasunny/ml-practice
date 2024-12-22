import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from features.create_features import create_features


def main():
    # サンプルデータの作成
    symbol = "7203.T"  # トヨタ自動車の例
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    daily_data = fetch_stock_data(symbol, start_date, end_date)

    # ラベルデータの作成
    labeled_data = create_labels(daily_data)

    # 取引開始日とデータ数の設定
    trade_start_date = pd.Timestamp("2020-06-01")
    data_numbers = 2

    # 特徴量の作成
    try:
        features_df, features_all_df = create_features(
            daily_data, trade_start_date, labeled_data, data_numbers
        )

        # 結果の表示
        print(f"len(features_df) : {len(features_df)}")
        print("サンプリングデータの特徴量:")
        print(features_df.head())
        print(f"\nlen(features_all_df) : {len(features_all_df)}")
        print("全データの特徴量:")
        print(features_all_df.head())

        # 追加表示
        print("\n特徴量の詳細:")
        print("\n--- サンプリングデータの統計情報 ---")
        print(features_df.describe())
        print("\n--- 全データの統計情報 ---")
        print(features_all_df.describe())

        # 不正解ラベルのデータ数と正解ラベルのデータ数を表示
        label_counts = features_df["Label"].value_counts()
        print(f"不正解ラベル数: {label_counts[0]}")
        print(f"正解ラベル数  : {label_counts[1]}")

    except ValueError as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
