import pandas as pd
from utils.fetch_and_prepare_data import fetch_and_prepare_data
from model_training.data_preparation import prepare_data
from utils.run_model import run_model
from ensemble_methods.evaluate_ensemble import evaluate_ensemble  # 追加

# Pandasの表示オプションを設定 Noneで全て表示
pd.set_option("display.max_rows", 20)  # 表示する最大行数
pd.set_option("display.max_columns", 20)  # 表示する最大列数
pd.set_option("display.max_colwidth", 7)  # 各列の幅（文字数）
pd.set_option("display.width", None)  # 表示幅


def main():
    sector_number = "test"  # "csv" or "all", 1, 2, 3, 4, 5, "test" 数字は文字列でもOK
    trade_start_date = pd.Timestamp("2005-08-01")
    before_period_days = 366 * 3
    end_date = pd.Timestamp("today")
    data_numbers = 1  # features_df生成時の正解ラベルに対する不正解ラベルの倍数制限

    # --------------------------データ取得、学習データ、特徴量、ラベルの生成

    features_df_for_train, features_df_for_evaluation, symbol_data_dict = (
        fetch_and_prepare_data(
            sector_number, trade_start_date, before_period_days, end_date, data_numbers
        )
    )
    training_features_df = features_df_for_train.drop(columns=["Symbol"])
    predict_features_df = features_df_for_evaluation.drop(columns=["Symbol"])

    # --------------------------トレインデータの準備
    X_train, X_test, y_train, y_test = prepare_data(training_features_df)

    # --------------------------モデルのトレーニングと評価
    model_types = [
        "lightgbm",
        "rand_frst",
        "xgboost",
        "catboost",
        "adaboost",
        "grdt_bstg",
        "svm",
        "knn",
        "logc_regr",
    ]

    # 結果を保存するための辞書を定義
    results = {}
    all_symbol_signals = {}

    for model_type in model_types:
        stock_response, symbol_signals = run_model(
            model_type,
            X_train,
            X_test,
            y_train,
            y_test,  # トレーニング評価用の正解ラベル
            predict_features_df,  # 実践用正解ラベルはここから抽出する
            features_df_for_evaluation,
            symbol_data_dict,
            results,
        )
        all_symbol_signals[model_type] = symbol_signals

    # データフレームを作成（転置しない）
    results_df = pd.DataFrame(results).T

    print("*･゜ﾟ･*:.｡..｡.:*･゜最終(n^--^)結果ηﾟ･*:.｡. .｡.:*･゜ﾟ･*")
    print(results_df)

    # アンサンブル評価を実行
    print("ﾝ―((･ω｀･；三；･´ω･))―ﾝ アンサンブル評価を実行 ｩ──σ(´･д･`; )──ﾝ")

    evaluate_ensemble(all_symbol_signals, predict_features_df, symbol_data_dict)


if __name__ == "__main__":
    main()
