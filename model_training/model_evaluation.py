import pandas as pd
from model_training.plot_buy_chart import plot_results
from sklearn.metrics import confusion_matrix


def model_predict_and_plot(
    gbm, training_features_df, all_features_df, symbol_data_dict
):
    # モデルの予測と結果の確認
    X_test = training_features_df.drop("Label", axis=1)
    y_test = training_features_df["Label"]
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 結果をデータフレームにまとめ、シンボルカラムを追加して確認
    results_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred_binary,
            "Symbol": all_features_df["Symbol"],
        },
        index=training_features_df.index,
    )

    # 予測が外れた部分の表示
    # incorrect_predictions = results_df[results_df["Actual"] != results_df["Predicted"]]
    # print("予測が外れた部分:")
    # print(incorrect_predictions)

    # カスタムの評価を呼び出す
    recall, not_recall, accuracy, precision, f1_score, npv = custom_metrics(
        y_test, y_pred_binary
    )

    print(f"Accuracy [(TP + TN) / TT]  : {accuracy:.4f}")
    print(f"Precision [TP / (TP + FP)] : {precision:.4f}")
    print(f"Recall [TP / (TP + FN)]    : {recall:.4f}")
    print(f"Not-Recall [TN / (TN + FP)]: {not_recall:.4f}")
    print(f"F1 Score [2 * (Precision * Recall) / (Precision + Recall)]: {f1_score:.4f}")
    print(f"NPV [TN / (TN + FN)]       : {npv:.4f}")

    # # 各シンボルごとのプロット
    # for symbol in symbol_data_dict.keys():
    #     daily_data = symbol_data_dict[symbol]
    #     features_df = all_features_df[all_features_df["Symbol"] == symbol]
    #     plot_results(daily_data, features_df, results_df, symbol)


def custom_metrics(y_test, y_pred_binary):
    # Recallのデバック
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
