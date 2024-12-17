# main.py
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
from features.create_features import create_features
from model_training.model_evaluation import model_predict_and_plot
from sectors import get_symbols_by_sector  # セクターリストの関数をインポート
import pandas as pd
import time


def main():
    start_time = time.time()  # 処理開始時刻を記録

    # セクター番号を指定
    # 1: 自動車セクター, 2: テクノロジーセクター, 3: 金融セクター, 4: 医薬品セクター, 5: 食品セクター
    sector_number = 3  # ここでセクター番号を指定してください

    # セクター番号に基づいて銘柄リストを取得
    symbols = get_symbols_by_sector(sector_number)

    trade_start_date = pd.Timestamp("2005-08-01")  # テスト用データの最初の日
    before_period_days = 366 * 3  # 月足取得に必要な期間（月足36個にしている約3年分）
    end_date = pd.Timestamp("today")  # 最新の日付に設定

    all_features_df = pd.DataFrame()
    all_features_all_df = pd.DataFrame()
    symbol_data_dict = {}  # 銘柄ごとのデータを格納する辞書

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
            data_numbers = 4  # 不正解データが正解ラベルの 4倍
            features_df, features_all_df = create_features(
                daily_data, trade_start_date, labeled_data, data_numbers
            )

            # シンボルカラムを追加
            features_df["Symbol"] = symbol
            features_all_df["Symbol"] = symbol

            # 欠損値の削除
            features_df.dropna(inplace=True)
            features_all_df.dropna(inplace=True)

            # 結合
            all_features_df = pd.concat([all_features_df, features_df])
            all_features_all_df = pd.concat([all_features_all_df, features_all_df])

            # 各シンボルごとのデータを辞書に格納
            symbol_data_dict[symbol] = daily_data

            # タイムラグを設ける
            time.sleep(1)

        except Exception as e:
            print(f"エラーが発生しました: {symbol}, {e}")

    # 学習に使用する特徴量データフレームからシンボルカラムを除外
    training_features_df = all_features_df.drop(columns=["Symbol"])
    model_predict_features_df = all_features_all_df.drop(columns=["Symbol"])

    # 処理時間
    end_time = time.time()  # 処理終了時刻を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print(f"学習データ生成の処理時間: {elapsed_time:.2f}秒")  # 処理時間を表示

    # モデルの学習と評価
    gbm = train_and_evaluate_model(training_features_df)

    # モデルの予測と結果の確認
    model_predict_and_plot(
        gbm, model_predict_features_df, all_features_all_df, symbol_data_dict
    )


if __name__ == "__main__":
    main()
