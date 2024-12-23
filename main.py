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


def fetch_and_prepare_data(
    sector_number, trade_start_date, before_period_days, end_date, data_numbers
):
    symbols = get_symbols_by_sector(sector_number)
    all_features_df_for_train = pd.DataFrame()
    all_features_df_for_evaluate = pd.DataFrame()
    symbol_data_dict = {}

    for symbol in symbols:
        try:
            start_date = trade_start_date - pd.Timedelta(days=before_period_days)
            daily_data = fetch_stock_data(symbol, start_date, end_date)
            if daily_data.empty:
                print(f"データが見つかりませんでした: {symbol}")
                continue
            labeled_data = create_labels(daily_data)
            features_df_for_train, features_all_df_evaluate = create_features(
                daily_data, trade_start_date, labeled_data, data_numbers
            )
            features_df_for_train["Symbol"] = symbol
            features_all_df_evaluate["Symbol"] = symbol
            features_df_for_train.dropna(inplace=True)
            features_all_df_evaluate.dropna(inplace=True)
            all_features_df_for_train = pd.concat(
                [all_features_df_for_train, features_df_for_train]
            )
            all_features_df_for_evaluate = pd.concat(
                [all_features_df_for_evaluate, features_all_df_evaluate]
            )
            symbol_data_dict[symbol] = daily_data
            time.sleep(1)
        except Exception as e:
            print(f"エラーが発生しました: {symbol}, {e}")

    return all_features_df_for_train, all_features_df_for_evaluate, symbol_data_dict


def optimize_parameters_for_symbols(symbol_signals, symbol_data_dict):
    optimal_params = []
    least_optimal_params = []
    rejected_params = []

    for symbol, signals in symbol_signals.items():
        daily_data = symbol_data_dict[symbol]
        for signal_date in signals:
            best_result, worst_result, _ = optimize_parameters(daily_data, signal_date)
            if not any(
                param["stop_loss_percentage"] == best_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == best_result["trailing_stop_trigger"]
                and param["trailing_stop_update"] == best_result["trailing_stop_update"]
                for param in optimal_params
            ):
                optimal_params.append(best_result)
            else:
                rejected_params.append(best_result)
            if not any(
                param["stop_loss_percentage"] == worst_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == worst_result["trailing_stop_trigger"]
                and param["trailing_stop_update"]
                == worst_result["trailing_stop_update"]
                for param in least_optimal_params
            ):
                least_optimal_params.append(worst_result)
            else:
                rejected_params.append(worst_result)

    return symbol_signals, optimal_params, least_optimal_params, rejected_params


def main():
    sector_number = 3
    trade_start_date = pd.Timestamp("2005-08-01")
    before_period_days = 366 * 3
    end_date = pd.Timestamp("today")
    data_numbers = 3  # features_df生成時の正解ラベルに対する不正解ラベルの倍数制限

    # --------------------------データ取得、学習データ、特徴量、ラベルの生成
    start_time_features = time.time()
    features_df_for_train, features_df_for_evaluation, symbol_data_dict = (
        fetch_and_prepare_data(
            sector_number, trade_start_date, before_period_days, end_date, data_numbers
        )
    )
    training_features_df = features_df_for_train.drop(columns=["Symbol"])
    model_predict_features_df = features_df_for_evaluation.drop(columns=["Symbol"])
    end_time_features = time.time()
    print(
        f"データ取得、学習データ、特徴量、ラベルの生成 処理時間: {end_time_features - start_time_features:.2f}秒"
    )

    # --------------------------モデルのトレーニング
    start_time_train = time.time()
    gbm = train_and_evaluate_model(training_features_df)
    end_time_train = time.time()
    print(f"モデルのトレーニングの処理時間: {end_time_train - start_time_train:.2f}秒")

    # --------------------------トレーニング後のモデルの評価
    symbol_signals = model_predict(
        gbm, model_predict_features_df, features_df_for_evaluation, symbol_data_dict
    )

    # --------------------------以下は将来、別のプロジェクトにて、Goに移管したい
    # --------------------------銘柄毎の最適パラメータ抽出
    start_time_optimize = time.time()
    symbol_signals, optimal_params, least_optimal_params, rejected_params = (
        optimize_parameters_for_symbols(symbol_signals, symbol_data_dict)
    )
    end_time_optimize = time.time()
    print(
        f"銘柄毎の最適パラメータ抽出の処理時間: {end_time_optimize - start_time_optimize:.2f}秒"
    )
    # --------------------------セクター全体の最適パラメータ探索
    start_time_best = time.time()
    best_params, max_profit_loss, param_results = find_best_params(
        optimal_params, symbol_data_dict, symbol_signals, trading_strategy
    )
    display_params(best_params, max_profit_loss, param_results, "良い : 最適な群で")
    end_time_best = time.time()
    print(
        f"セクター全体(銘柄数:{len(symbol_signals)})の最適パラメータ探索の処理時間: {end_time_best - start_time_best:.2f}秒"
    )
    start_time_least_optimal = time.time()
    least_optimal_params_best, min_profit_loss, least_param_results = find_worst_params(
        least_optimal_params, symbol_data_dict, symbol_signals, trading_strategy
    )
    display_params(
        least_optimal_params_best,
        min_profit_loss,
        least_param_results,
        "悪い : 最適から遠い群で",
    )
    end_time_least_optimal = time.time()
    print(
        f"最適でないパラメータ探索の処理時間: {end_time_least_optimal - start_time_least_optimal:.2f}秒"
    )


if __name__ == "__main__":
    main()
