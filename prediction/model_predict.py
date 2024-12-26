import pandas as pd
from prediction.evaluate_prediction import evaluate_metrics
from model_training.plot_buy_chart import plot_results
import numpy as np


def model_predict(
    model, model_predict_features_df, features_df_for_evaluation, symbol_data_dict
):
    # モデルの予測と結果の確認
    X_test = model_predict_features_df.drop("Label", axis=1)  # 説明変数
    y_test = model_predict_features_df["Label"]  # 目的変数

    # モデルが predict_proba を持っているか確認
    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]  # 確率を予測
    else:
        y_pred = model.predict(X_test)  # バイナリ予測

    y_pred_binary = (y_pred > 0.5).astype(int)

    # 結果をデータフレームにまとめ、シンボルカラムを追加して確認
    results_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred_binary,
            "Symbol": features_df_for_evaluation["Symbol"],
        },
        index=model_predict_features_df.index,
    )

    # カスタムの評価を呼び出す
    accuracy, precision, recall, not_recall, f1_score, npv = evaluate_metrics(
        y_test, y_pred_binary
    )

    symbol_signals = {}  # シンボルごとの予測結果を格納する辞書

    # 各シンボルごとの予測結果を抽出して辞書に格納
    for symbol in symbol_data_dict.keys():
        daily_data = symbol_data_dict[symbol]
        features_df = features_df_for_evaluation[
            features_df_for_evaluation["Symbol"] == symbol
        ]
        symbol_signals[symbol] = results_df[
            (results_df["Symbol"] == symbol) & (results_df["Predicted"] == 1)
        ].index
        # plot_results(daily_data, features_df, results_df, symbol)  # 目視で確認、重要！

    # 評価指標を追加して返す
    tp = np.sum((y_pred_binary == 1) & (y_test == 1))
    tn = np.sum((y_pred_binary == 0) & (y_test == 0))
    fp = np.sum((y_pred_binary == 1) & (y_test == 0))
    fn = np.sum((y_pred_binary == 0) & (y_test == 1))
    total_tests = len(y_test)

    return (
        symbol_signals,
        tp,
        tn,
        fp,
        fn,
        total_tests,
        accuracy,
        precision,
        recall,
        not_recall,
        f1_score,
        npv,
    )


# print(f"symbol_signals : {symbol_signals}")
# symbol_signalsの構造 (例 knnモデル)
# symbol_signals : {'8306.T': DatetimeIndex(['2015-01-19', '2011-03-17', '2017-12-07', '2023-07-24',
#                '2013-11-08', '2019-08-26', '2008-12-08', '2012-07-25',
#                '2016-11-09', '2009-07-13',
#                ...
#                '2024-10-29', '2024-10-30', '2024-10-31', '2024-11-05',
#                '2024-12-17', '2024-12-18', '2024-12-19', '2024-12-24',
#                '2024-12-25', '2024-12-26'],
#               dtype='datetime64[ns]', length=2516, freq=None)}
