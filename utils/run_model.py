import time
from model_training.train_models import train_model
from prediction.model_predict import model_predict
from model_io.save_model import save_model
from model_io.load_model import load_model
from proto_definitions.proto_conversion import convert_to_proto_response


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

    print(f"★ モデル ★{model_type}★ による予測完了 ★")

    # symbol_signalsをProtoBufのレスポンスデータ構造に変換
    stock_response = convert_to_proto_response(symbol_signals, symbol_data_dict)

    return stock_response


# print(f"symbol_data_dict : {symbol_data_dict}")
# symbol_signalsの構造 (例 knnモデル)
# symbol_data_dict : {'8306.T':            Open    High     Low   Close  Adj Close  Volume  Label
# Date
# 2005...  1460.0  1480.0  1410.0  1450.0  831...     819...      0
# 2005...  1460.0  1530.0  1420.0  1490.0  854...     918...      0
# 2005...  1460.0  1470.0  1380.0  1420.0  814...     116...      0
# 2005...  1430.0  1440.0  1380.0  1400.0  802...     473...      0
# 2005...  1410.0  1440.0  1380.0  1400.0  802...     718...      0
# ...         ...     ...     ...     ...     ...        ...    ...
# 2024...  1815.0  1819.5  1768.5  1773.0  177...     707...      1
# 2024...  1774.0  1800.0  1768.5  1800.0  180...     353...      0
# 2024...  1808.0  1815.5  1799.0  1808.0  180...     295...      0
# 2024...  1801.5  1805.0  1788.0  1800.0  180...     253...      0
# 2024...  1800.0  1804.0  1793.0  1798.0  179...     821...      0

# [4749 rows x 7 columns]}
