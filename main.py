from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
from features.create_features import create_features
from setting_stop.optimize_parameters import optimize_parameters
from setting_stop.trading_strategy import trading_strategy
from setting_stop.plot_heatmap import plot_heatmap
from setting_stop.print_results import print_results  # 新しいファイルをインポート
from sectors import get_symbols_by_sector  # セクターリストの関数をインポート
import pandas as pd
import numpy as np
import time

def main():
    start_time_features = time.time()  # 処理開始時刻を記録

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
    end_time_features = time.time()  # 処理終了時刻を記録
    elapsed_time_features = end_time_features - start_time_features  # 経過時間を計算
    print(f"学習データ生成の処理時間: {elapsed_time_features:.2f}秒")  # 処理時間を表示


    # ------------------------------------------------------------------------------

    # モデルの学習と評価
    gbm = train_and_evaluate_model(training_features_df)

    # ------------------------------------------------------------------------------


    start_time_optimize = time.time()  # 処理開始時刻を記録

    # 遅延インポート
    from model_training.model_evaluation import model_predict

    # モデルの予測とシンボル毎の結果の取得
    symbol_signals = model_predict(
        gbm, model_predict_features_df, all_features_all_df, symbol_data_dict
    )
    print(f"symbol_signals : {len(symbol_signals)}")
   
    optimal_params = []  # 最適解群の配列を用意する
    rejected_params = []  # 重複して排除されたパラメータを格納するリスト

    # ストップオーダーのパラメータ最適化の実行
    for symbol, signals in symbol_signals.items():
        daily_data = symbol_data_dict[symbol]  # シンボルに紐づいた株価データを取り出し
        for signal_date in signals:  # daily_dataに紐づいたsymbol毎のsignalを抽出
            best_result, _ = optimize_parameters(daily_data, signal_date)
            
            # 重複チェック
            is_duplicate = any(
                param["stop_loss_percentage"] == best_result["stop_loss_percentage"] and
                param["trailing_stop_trigger"] == best_result["trailing_stop_trigger"] and
                param["trailing_stop_update"] == best_result["trailing_stop_update"]
                for param in optimal_params
            )
            
            # 重複していない場合のみ追加、重複している場合は排除リストに追加
            if not is_duplicate:
                optimal_params.append(best_result)  # 最適解群の配列に最適解を入れる
            else:
                rejected_params.append(best_result)  # 重複して排除されたパラメータを格納

    # 重複して排除されたパラメータを表示
    # print("重複して排除されたパラメータ:")
    # for param in rejected_params:
    #     print(f"損失割合: {param['stop_loss_percentage']}%, トレーリングストップトリガー: {param['trailing_stop_trigger']}%, トレーリングストップ更新: {param['trailing_stop_update']}%")

    # 最適解群の中から、全ての銘柄の全てのシグナルにエントリーした際、最も損益が高くなるストップオーダーのパラメータを見つける
    best_params = None
    max_profit_loss = -np.inf  # 損益の初期値を最小値に設定

    print(f"銘柄毎の最適パラメータ解群からベスト解の探索開始")
    for params in optimal_params:  # パラメータを抽出
        sum_profit_loss = 0  # 各パラメータごとに総損益を計算
        for symbol, daily_data in symbol_data_dict.items():
            for signal_date in symbol_signals[symbol]:  # daily_dataに紐づいたsymbol毎のsignalを抽出
                _, _, _, _, profit_loss = trading_strategy(
                    daily_data.copy(),
                    signal_date,
                    params["stop_loss_percentage"],
                    params["trailing_stop_trigger"],
                    params["trailing_stop_update"],
                )
                sum_profit_loss += profit_loss
                # print(f"profit_loss {profit_loss} , sum_profit_loss += profit_loss : {sum_profit_loss}")
            
        # 損益が現在の最大値を上回る場合、パラメータを更新
        if sum_profit_loss > max_profit_loss:
            max_profit_loss = sum_profit_loss
            best_params = params

    print(f"総損益: {max_profit_loss:.2f}%")
    if best_params is not None and not best_params.empty:
        print("最も損益が高かったストップオーダーのパラメータ:")
        print(f"初回LC値: {best_params['stop_loss_percentage']}%")
        print(f"TSトリガー値: {best_params['trailing_stop_trigger']}%")
        print(f"TS更新値: {best_params['trailing_stop_update']}%")
    else:
        print("最適なパラメータが見つかりませんでした。")

    # 処理時間
    end_time_optimize = time.time()  # 処理終了時刻を記録
    elapsed_time_optimize = end_time_optimize - start_time_optimize  # 経過時間を計算
    print(f"最適パラメータ探索の処理時間: {elapsed_time_optimize:.2f}秒")  # 処理時間を表示

if __name__ == "__main__":
    main()
