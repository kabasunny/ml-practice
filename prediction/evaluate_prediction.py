from sklearn.metrics import confusion_matrix


def evaluate_metrics(y_test, y_pred_binary):

    print("<----------------実践条件での評価結果----------------->")

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
    f1_score = 2 * (precision * recall) / (precision + recall)  # F1 Score
    npv = TN / (TN + FN)  # Negative Predictive Value（陰性的中率）

    print(f"Accuracy [(TP + TN) / TT]  : {accuracy:.4f}")
    print(f"Precision [TP / (TP + FP)] : {precision:.4f}")
    print(f"Recall [TP / (TP + FN)]    : {recall:.4f}")
    print(f"Not-Recall [TN / (TN + FP)]: {not_recall:.4f}")
    print(f"F1 Score [2 * (Precision * Recall) / (Precision + Recall)]: {f1_score:.4f}")
    print(f"NPV [TN / (TN + FN)]       : {npv:.4f}")

    return accuracy, precision, recall, not_recall, f1_score, npv
