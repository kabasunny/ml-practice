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

    # 結果をフォーマット
    formatted_results = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "T_Tests": int(total_tests),
        "Accuracy": f"{accuracy:.3f}",
        "Precision": f"{precision:.3f}",
        "Recall": f"{recall:.3f}",
        "Not-Recall": f"{not_recall:.3f}",
        "F1 Score": f"{f1_score:.3f}",
    }

    # 結果を保存
    results[model_type] = formatted_results

    print(f"モデル {model_type} による予測完了")
