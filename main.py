import pandas as pd
import time
from utils.fetch_and_prepare_data import fetch_and_prepare_data
from model_training.data_preparation import prepare_data
from model_training.train_lightgbm import train_lightgbm
from prediction.model_predict import model_predict
import joblib


# モデルの保存
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"モデルを {filename} に保存しました")


# モデルの読み込み
def load_model(filename):
    model = joblib.load(filename)
    print(f"モデルを {filename} から読み込みました")
    return model


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

    # --------------------------データの準備
    X_train, X_test, y_train, y_test = prepare_data(training_features_df)

    # --------------------------モデルのトレーニング
    start_time_train = time.time()
    gbm = train_lightgbm(X_train, X_test, y_train, y_test)
    end_time_train = time.time()
    print(f"モデルのトレーニングの処理時間: {end_time_train - start_time_train:.2f}秒")

    # モデルを保存
    save_model(gbm, "trained_model.pkl")

    # --------------------------保存されたモデルを読み込んで評価
    loaded_model = load_model("trained_model.pkl")
    symbol_signals = model_predict(
        loaded_model,
        model_predict_features_df,
        features_df_for_evaluation,
        symbol_data_dict,
    )


if __name__ == "__main__":
    main()


# # --------------------------以下は将来、別のプロジェクトにて、Goに移管したい

# from params_search.best_params import find_best_params
# from params_search.worst_params import find_worst_params
# from params_search.display_params import display_params
# from setting_stop.trading_strategy import trading_strategy
# from utils.optimize_parameters_for_symbols import optimize_parameters_for_symbols

# # --------------------------銘柄毎の最適パラメータ抽出
# start_time_optimize = time.time()
# symbol_signals, optimal_params, least_optimal_params, rejected_params = (
#     optimize_parameters_for_symbols(symbol_signals, symbol_data_dict)
# )
# end_time_optimize = time.time()
# print(
#     f"銘柄毎の最適パラメータ抽出の処理時間: {end_time_optimize - start_time_optimize:.2f}秒"
# )


#     # --------------------------セクター全体の最適パラメータ探索
#     start_time_best = time.time()
#     best_params, max_profit_loss, param_results = find_best_params(
#         optimal_params, symbol_data_dict, symbol_signals, trading_strategy
#     )
#     display_params(best_params, max_profit_loss, param_results, "良い : 最適な群で")
#     end_time_best = time.time()
#     print(
#         f"セクター全体(銘柄数:{len(symbol_signals)})の最適パラメータ探索の処理時間: {end_time_best - start_time_best:.2f}秒"
#     )

#     start_time_least_optimal = time.time()
#     least_optimal_params_best, min_profit_loss, least_param_results = find_worst_params(
#         least_optimal_params, symbol_data_dict, symbol_signals, trading_strategy
#     )
#     display_params(
#         least_optimal_params_best,
#         min_profit_loss,
#         least_param_results,
#         "悪い : 最適から遠い群で",
#     )
#     end_time_least_optimal = time.time()
#     print(
#         f"最適でないパラメータ探索の処理時間: {end_time_least_optimal - start_time_least_optimal:.2f}秒"
#     )


# if __name__ == "__main__":
#     main()
