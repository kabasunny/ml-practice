from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    # テストデータに対する予測を行い、0と1に変換
    y_pred = model.predict(X_test)

    if hasattr(y_pred, "shape") and len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)  # One-hot encoded outputの場合
    else:
        y_pred = (y_pred > 0.5).astype(int)  # バイナリ出力の場合

    # モデルの評価指標を計算
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 評価結果を表示
    print("||||  トレーニング修了時の一般的な評価結果  ||||")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # return accuracy, precision, recall, f1
