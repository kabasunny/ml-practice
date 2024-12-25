import pandas as pd
import time
from utils.fetch_and_prepare_data import fetch_and_prepare_data
from model_training.data_preparation import prepare_data
from utils.run_model import run_model  # run_model 関数をインポート
from model_io.save_model import save_model
from model_io.load_model import load_model

# Pandasの表示オプションを設定
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


def main():
    sector_number = 3
    trade_start_date = pd.Timestamp("2005-08-01")
    before_period_days = 366 * 3
    end_date = pd.Timestamp("today")
    data_numbers = 2  # features_df生成時の正解ラベルに対する不正解ラベルの倍数制限

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

    model_types = [
        "lightgbm",
        "random_forest",
        "xgboost",
        "catboost",
        "adaboost",
        "gradient_boosting",
        "svm",
        "knn",
        "logistic_regression",
    ]

    # 結果を保存するための辞書を定義
    results = {}

    for model_type in model_types:
        run_model(
            model_type,
            X_train,
            X_test,
            y_train,
            y_test,
            model_predict_features_df,
            features_df_for_evaluation,
            symbol_data_dict,
            results,
        )

    results_df = pd.DataFrame(results).T
    results_df = results_df.transpose()  # 縦横を入れ替え
    print("\n最終結果:")
    print(results_df)


if __name__ == "__main__":
    main()
