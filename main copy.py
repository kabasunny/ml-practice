from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
from model_training.model_evaluation import model_predict
from features.create_features import create_features
from setting_stop.optimize_parameters import optimize_parameters
from setting_stop.trading_strategy import trading_strategy
from params_serch.best_params import find_best_params
from params_serch.worst_params import find_worst_params
from params_serch.display_params import display_params
from sectors import get_symbols_by_sector
import pandas as pd
import time


def main():
    start_time_features = time.time()  # 処理開始時刻を記録
    print(f"データ取得、学習データ、特徴量、ラベルの生成 開始")

    # セクター番号を指定
    # 1: 自動車セクター, 2: テクノロジーセクター, 3: 金融セクター, 4: 医薬品セクター, 5: 食品セクター
    sector_number = 1  # ここでセクター番号を指定してください

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
            data_numbers = 2  # 不正解データが正解ラベルの 4倍
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
    print(
        f"データ取得、学習データ、特徴量、ラベルの生成 処理時間: {elapsed_time_features:.2f}秒"
    )  # 処理時間を表示

    # ------------------------------------------------------------------------------

    start_time_train = time.time()  # 処理開始時刻を記録

    # モデルの学習と評価
    gbm = train_and_evaluate_model(training_features_df)

    # 処理時間
    end_time_train = time.time()  # 処理終了時刻を記録
    elapsed_time_train = end_time_train - start_time_train  # 経過時間を計算
    print(
        f"モデルのトレーニングの処理時間: {elapsed_time_train:.2f}秒"
    )  # 処理時間を表示

    # ------------------------------------------------------------------------------

    start_time_optimize = time.time()  # 処理開始時刻を記録

    # モデルの予測とシンボル毎の結果の取得
    symbol_signals = model_predict(
        gbm, model_predict_features_df, all_features_all_df, symbol_data_dict
    )
    print(f"銘柄数 : {len(symbol_signals)}, 各最適パラメータ抽出開始")
    optimal_params = []  # 最適解群の配列を用意する
    least_optimal_params = []  # 最悪解群の配列を用意する
    rejected_params = []  # 重複して排除されたパラメータを格納するリスト
    count_symbol = 0

    # ストップオーダーのパラメータ最適化の実行
    for symbol, signals in symbol_signals.items():
        count_symbol += 1
        print(f"{count_symbol}番目の銘柄の探索中, {len(signals)}個のシグナルで最適化")
        daily_data = symbol_data_dict[symbol]  # シンボルに紐づいた株価データを取り出し
        for signal_date in signals:  # daily_dataに紐づいたsymbol毎のsignalを抽出
            best_result, worst_result, _ = optimize_parameters(daily_data, signal_date)

            # 重複チェック
            is_duplicate_best = any(
                param["stop_loss_percentage"] == best_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == best_result["trailing_stop_trigger"]
                and param["trailing_stop_update"] == best_result["trailing_stop_update"]
                for param in optimal_params
            )

            is_duplicate_worst = any(
                param["stop_loss_percentage"] == worst_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == worst_result["trailing_stop_trigger"]
                and param["trailing_stop_update"]
                == worst_result["trailing_stop_update"]
                for param in least_optimal_params
            )

            # 重複していない場合のみ追加、重複している場合は排除リストに追加
            if not is_duplicate_best:
                optimal_params.append(best_result)  # 最適解群の配列に最適解を入れる
            else:
                rejected_params.append(
                    best_result
                )  # 重複して排除されたパラメータを格納

            if not is_duplicate_worst:
                least_optimal_params.append(
                    worst_result
                )  # 最悪解群の配列に最悪解を入れる
            else:
                rejected_params.append(
                    worst_result
                )  # 重複して排除されたパラメータを格納

    # 最適化パラメータを表示
    print(f'最適化パラメータ数 : {len(optimal_params)}')

    # 最適化パラメータを表示
    print(f'最適から遠いパラメータ数 : {len(least_optimal_params)}')

    # 重複して排除されたパラメータを表示
    print(f'重複して排除されたパラメータ数 : {len(rejected_params)}')
    # for param in rejected_params:
    #     print(f"LC: {param['stop_loss_percentage']}%, TSトリガー: {param['trailing_stop_trigger']}%, TS更新: {param['trailing_stop_update']}%")

    # 処理時間
    end_time_optimize = time.time()  # 処理終了時刻を記録
    elapsed_time_optimize = end_time_optimize - start_time_optimize  # 経過時間を計算
    print(
        f"銘柄毎の最適パラメータ抽出の処理時間: {elapsed_time_optimize:.2f}秒"
    )  # 処理時間を表示
    # ------------------------------------------------------------------------------

    start_time_best = time.time()  # 処理開始時刻を記録
    # 最適解群の中から、全ての銘柄の全てのシグナルにエントリーした際、最も損益が高くなるストップオーダーのパラメータを見つける
    print(
        f"len(optimal_params)*len(symbol_data_dict)*len(symbol_signals): {len(optimal_params)*len(symbol_data_dict)*len(symbol_signals)}"
    )

    best_params, max_profit_loss, param_results = find_best_params(
        optimal_params, symbol_data_dict, symbol_signals, trading_strategy
    )

    display_params(best_params, max_profit_loss, param_results, "良い : 最適な群で")

    # 処理時間
    end_time_best = time.time()  # 処理終了時刻を記録
    elapsed_time_best = end_time_best - start_time_best  # 経過時間を計算
    print(
        f"セクター全体(銘柄数:{len(symbol_signals)})の最適パラメータ探索の処理時間: {elapsed_time_best:.2f}秒"
    )  # 処理時間を表示

    # ------------------------------------------------------------------------------

    start_time_least_optimal = time.time()  # 処理開始時刻を記録
    # 最適でない解群の中から、全ての銘柄の全てのシグナルにエントリーした際、最も損益が低くなるストップオーダーのパラメータを見つける
    print(
        f"len(least_optimal_params)*len(symbol_data_dict)*len(symbol_signals): {len(least_optimal_params)*len(symbol_data_dict)*len(symbol_signals)}"
    )

    least_optimal_params_best, min_profit_loss, least_param_results = find_worst_params(
        least_optimal_params, symbol_data_dict, symbol_signals, trading_strategy
    )

    display_params(
        least_optimal_params_best,
        min_profit_loss,
        least_param_results,
        "悪い : 最適から遠い群で",
    )

    # 処理時間
    end_time_least_optimal = time.time()  # 処理終了時刻を記録
    elapsed_time_least_optimal = (
        end_time_least_optimal - start_time_least_optimal
    )  # 経過時間を計算
    print(
        f"最適でないパラメータ探索の処理時間: {elapsed_time_least_optimal:.2f}秒"
    )  # 処理時間を表示

    # ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
