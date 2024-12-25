import time
from model_training.train_models import train_model
from prediction.model_predict import model_predict
from model_io.save_model import save_model
from model_io.load_model import load_model


def run_model(
    model_type,
    X_train,
    X_test,
    y_train,
    y_test,
    model_predict_features_df,
    features_df_for_evaluation,
    symbol_data_dict,
    results,
):
    print("----------------------------------------")
    print(f"実行中のモデルタイプ: {model_type}")

    # --------------------------モデルのトレーニング
    start_time_train = time.time()
    model = train_model(model_type, X_train, X_test, y_train, y_test)
    end_time_train = time.time()
    print(f"モデルのトレーニングの処理時間: {end_time_train - start_time_train:.2f}秒")

    # モデルを保存
    save_model(model, model_type)

    # --------------------------保存されたモデルを読み込んで評価
    loaded_model = load_model(model_type)
    (
        _,
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
        _,
    ) = model_predict(
        loaded_model,
        model_predict_features_df,
        features_df_for_evaluation,
        symbol_data_dict,
    )

    # 途中結果を表示
    print(f"モデル {model_type} による予測結果:")
    print(f"True Positives  (TP) : {tp}")
    print(f"True Negatives  (TN) : {tn}")
    print(f"False Positives (FP) : {fp}")
    print(f"False Negatives (FN) : {fn}")
    print(f"Total Tests     (TT) : {total_tests}")
    print(f"Accuracy [(TP + TN) / TT]  : {accuracy:.4f}")
    print(f"Precision [TP / (TP + FP)] : {precision:.4f}")
    print(f"Recall [TP / (TP + FN)]    : {recall:.4f}")
    print(f"Not-Recall [TN / (TN + FP)]: {not_recall:.4f}")
    print(f"F1 Score [2 * (Precision * Recall) / (Precision + Recall)]: {f1_score:.4f}")

    # 結果を保存
    results[model_type] = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Total Tests": total_tests,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Not-Recall": not_recall,
        "F1 Score": f1_score,
    }

    print(f"モデル {model_type} による予測完了")
