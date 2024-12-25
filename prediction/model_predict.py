import pandas as pd
from prediction.evaluate_prediction import evaluate_metrics


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

    print(f"Accuracy [(TP + TN) / TT]  : {accuracy:.4f}")
    print(f"Precision [TP / (TP + FP)] : {precision:.4f}")
    print(f"Recall [TP / (TP + FN)]    : {recall:.4f}")
    print(f"Not-Recall [TN / (TN + FP)]: {not_recall:.4f}")
    print(f"F1 Score [2 * (Precision * Recall) / (Precision + Recall)]: {f1_score:.4f}")
    print(f"NPV [TN / (TN + FN)]       : {npv:.4f}")

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

    return symbol_signals  # シンボルごとの予測結果を返す
