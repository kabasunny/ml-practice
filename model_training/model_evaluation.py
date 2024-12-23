import pandas as pd
from model_training.plot_buy_chart import plot_results
from sklearn.metrics import confusion_matrix


def model_predict(
    gbm, model_predict_features_df, features_df_for_evaluation, symbol_data_dict
):
    # モデルの予測と結果の確認
    X_test = model_predict_features_df.drop("Label", axis=1)  # 説明変数
    y_test = model_predict_features_df["Label"]  # 目的変数
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
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
    accuracy, precision, recall, not_recall, f1_score, npv = custom_metrics(
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
        plot_results(daily_data, features_df, results_df, symbol)  # 目視で確認、重要！

    return symbol_signals  # シンボルごとの予測結果を返す


# 戻り値の例
# {
#     'AAPL': Index(['2023-01-01', '2023-01-05', '2023-01-10'], dtype='datetime64[ns]'),
#     'GOOGL': Index(['2023-02-01', '2023-02-05'], dtype='datetime64[ns]'),
#     // 他のシンボル...
# }


def custom_metrics(y_test, y_pred_binary):
    # Recallのデバッグ
    print(f"model_evaluation.py [len(y_test):{len(y_test)}]")
    cm = confusion_matrix(y_test, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()
    print(f"True Positives  (TP) : {TP}")
    print(f"True Negatives  (TN) : {TN}")
    print(f"False Positives (FP) : {FP}")
    print(f"False Negatives (FN) : {FN}")
    print(f"Total Tests     (TT) : {len(y_test)}")

    accuracy = (TP + TN) / len(y_test)  # 一致数／テスト総数
    precision = TP / (TP + FP)  # Precision（適合率）
    recall = TP / (TP + FN)  # 正解ラベルに対する一致数／正解ラベル総数（Recall）
    not_recall = TN / (TN + FP)  # 不正解ラベルに対する一致数／不正解ラベル総数
    f1_score = (
        2 * (precision * recall) / (precision + recall)
    )  # F1 Score Precision（適合率）と Recall（再現率）の調和平均で、これら二つの指標をバランスよく評価するためのもの
    npv = TN / (TN + FN)  # Negative Predictive Value（陰性的中率）

    return accuracy, precision, recall, not_recall, f1_score, npv
