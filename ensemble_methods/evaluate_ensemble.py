from prediction.evaluate_prediction import evaluate_metrics
from ensemble_methods.extract_duplicate_values import extract_duplicate_values
from ensemble_methods.convert_to_binary_predictions import convert_to_binary_predictions
from itertools import combinations


def evaluate_ensemble(all_symbol_signals, model_predict_features_df):
    """
    複数のモデルの予測結果をアンサンブル評価する関数
    :param all_symbol_signals: 各モデルの予測結果を含む辞書
    :param model_predict_features_df: モデルの予測特徴データフレーム
    :return: アンサンブル予測結果
    """
    ensemble_results = {}

    # モデルの組み合わせを生成
    model_types = list(all_symbol_signals.keys())
    model_combinations = [comb for comb in combinations(model_types, 8)]

    for combination in model_combinations:
        print(f"★モデルの組み合わせ: {combination}★")

        print(f"len(all_symbol_signals) : {len(all_symbol_signals)}")
        print(f"len(model_predict_features_df) : {len(model_predict_features_df)}")

        # 現在のモデルの組み合わせに基づいて重複する値を抽出
        selected_signals = {model: all_symbol_signals[model] for model in combination}
        
        min_overlap_count = 2  # 重複する日付の最小回数
        duplicated_values = extract_duplicate_values(selected_signals, min_overlap_count)
        

        # テストデータのインデックスを取得
        test_indices = model_predict_features_df.index
        y_test = model_predict_features_df["Label"]
        print(f"len(y_test) : {len(y_test)}")

        # 重複する日付をy_pred_binaryへ変換
        y_pred_binary = convert_to_binary_predictions(duplicated_values, test_indices)
        print(f"len(y_pred_binary) : {len(y_pred_binary)}")

        # 評価指標を計算
        evaluate_metrics(y_test, y_pred_binary)

    return ensemble_results
