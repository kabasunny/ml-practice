import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from features.create_features import create_features

def main():
    # サンプルデータの作成
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    close_prices = np.random.rand(500) * 100
    volume = np.random.randint(1000, 5000, size=500)
    daily_data = pd.DataFrame({"Close": close_prices, "Volume": volume}, index=dates)

    # ラベルデータの作成
    labels = np.random.randint(0, 2, size=500)
    labeled_data = pd.DataFrame({"Label": labels}, index=dates)

    # 取引開始日とデータ数の設定
    trade_start_date = pd.Timestamp("2020-06-01")
    data_numbers = 2

    # 特徴量の作成
    try:
        features_df, features_all_df = create_features(daily_data, trade_start_date, labeled_data, data_numbers)

        # 結果の表示
        print("サンプリングデータの特徴量:")
        print(features_df.head())
        print("\n全データの特徴量:")
        print(features_all_df.head())
    except ValueError as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
