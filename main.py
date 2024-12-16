from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
from features.create_features import create_features
import pandas as pd
import matplotlib.pyplot as plt
import time

def main():
    start_time = time.time()  # 処理開始時刻を記録

    # 自動車セクターの銘柄リスト
    symbols = [
    "7203.T",  # Toyota Motor Corporation
    "7201.T",  # Nissan Motor Co., Ltd.
    "7267.T",  # Honda Motor Co., Ltd.
    "7261.T",  # Mazda Motor Corporation
    "7269.T",  # Suzuki Motor Corporation
    # "7262.T",  # Mitsubishi Motors Corporation
    "7270.T"   # Subaru Corporation
    ]

    trade_start_date = pd.Timestamp("2005-08-01") # テスト用データの最初の日
    before_period_days = 366 * 3 # 月足取得に必要な期間（月足36個にしている約3年分）
    end_date = pd.Timestamp("today") # 最新の日付に設定

    all_features_df = pd.DataFrame()

    for symbol in symbols:
        try:
            # データの取得
            start_date = trade_start_date - pd.Timedelta(days=before_period_days)
            daily_data = fetch_stock_data(symbol, start_date, end_date)

            # データが空でないことを確認
            if daily_data.empty:
                print(f"データが見つかりませんでした: {symbol}")
                continue

            # インデックスを日時インデックスに変換
            daily_data.index = pd.to_datetime(daily_data.index)

            # ラベルの作成
            labeled_data = create_labels(daily_data)

            # 特徴量の作成
            data_numbers = 4 # 不正解データが正解ラベルの 4倍 
            features_df = create_features(daily_data, trade_start_date, labeled_data, data_numbers)
            
            # 欠損値の削除
            features_df.dropna(inplace=True)

            # 結合
            all_features_df = pd.concat([all_features_df, features_df])

            # タイムラグを設ける
            time.sleep(1)

        except Exception as e:
            print(f"エラーが発生しました: {symbol}, {e}")

    # ラベルのユニークな値とそのカウントを確認
    print(all_features_df['Label'].value_counts())
    print(all_features_df)

    # 処理時間
    end_time = time.time()  # 処理終了時刻を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"学習データ生成の処理時間: {elapsed_time:.2f}秒")  # 処理時間を表示

    # チャートを表示、正解ラベルにマーク
    # plot_buy_signals(daily_data, all_features_df, symbol)

    # モデルの学習と評価
    gbm = train_and_evaluate_model(all_features_df)


if __name__ == "__main__":
    main()
