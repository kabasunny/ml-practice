from prediction.evaluate_prediction import evaluate_metrics
from ensemble_methods.extract_duplicate_values import extract_duplicate_values
from ensemble_methods.convert_to_binary_predictions import convert_to_binary_predictions
from itertools import combinations
from ensemble_methods.plot_ensemble_results import plot_ensemble_results


def evaluate_ensemble(all_symbol_signals, model_predict_features_df, symbol_data_dict):
    """
    複数のモデルの予測結果をアンサンブル評価する関数
    :param all_symbol_signals: 各モデルの予測結果を含む辞書
    :param model_predict_features_df: モデルの予測特徴データフレーム
    :param symbol_data_dict: シンボルデータ辞書（日次データ）
    :return: Precisionの高いベスト5の組み合わせと予測データ
    """
    ensemble_results = []  # プロット用に、アンサンブル結果を格納するリスト
    precision_results = []

    # モデルの組み合わせを生成
    model_types = list(all_symbol_signals.keys())
    model_combinations = []
    for i in range(2, len(model_types) + 1):  # 2以上のサイズを指定
        model_combinations.extend([comb for comb in combinations(model_types, i)])

    for combination in model_combinations:
        # print(f"★モデルの組み合わせ★\n{combination}")

        # 現在のモデルの組み合わせに基づいて重複する値を抽出
        selected_signals = {model: all_symbol_signals[model] for model in combination}

        if len(combination) > 2:
            min_overlap_count = len(combination) - 1  # 重複する日付の最小回数
        else:
            min_overlap_count = len(combination)

        duplicated_values = extract_duplicate_values(
            selected_signals, min_overlap_count
        )

        # テストデータのインデックスを取得
        test_indices = model_predict_features_df.index
        y_test = model_predict_features_df["Label"]
        symbols = list(duplicated_values.keys())

        # 重複する日付をy_pred_binaryへ変換
        y_pred_binary = convert_to_binary_predictions(
            duplicated_values, test_indices, symbols
        )

        # 評価指標を計算
        metrics = evaluate_metrics(y_test, y_pred_binary)
        (
            accuracy,
            precision,
            recall,
            not_recall,
            f1_score,
            npv,
            TP,
            TN,
            FP,
            FN,
            total_tests,
        ) = metrics

        # 結果を保存
        precision_results.append((combination, metrics, duplicated_values))

    # Precisionの高い順にソートしてベスト5を抽出
    precision_results = sorted(precision_results, key=lambda x: x[1][1], reverse=True)[
        :10
    ]

    # ベスト5のモデルの組み合わせと予測データを表示
    for result in precision_results:
        combination, metrics, duplicated_values = result
        (
            accuracy,
            precision,
            recall,
            not_recall,
            f1_score,
            npv,
            TP,
            TN,
            FP,
            FN,
            total_tests,
        ) = metrics
        print(f"モデルの組み合わせ: {combination}")
        print(f"Precision: {precision:.4f}")
        print(f"True Positives  (TP): {TP:.0f}")
        print(f"True Negatives  (TN): {TN:.0f}")
        print(f"False Positives (FP): {FP:.0f}")
        print(f"False Negatives (FN): {FN:.0f}")
        print(f"Total Tests     (TT): {total_tests:.0f}")
        print(f"Accuracy [(TP + TN) / TT]: {accuracy:.4f}")
        print(f"Precision [TP / (TP + FP)]: {precision:.4f}")
        print(f"Recall [TP / (TP + FN)]: {recall:.4f}")
        print(f"Not-Recall [TN / (TN + FP)]: {not_recall:.4f}")
        print(
            f"F1 Score [2 * (Precision * Recall) / (Precision + Recall)]: {f1_score:.4f}"
        )
        print(f"NPV [TN / (TN + FN)]: {npv:.4f}")
        print("----------------------------------------------")

        # プロットも行う場合
        # for symbol in symbols:
        #     plot_ensemble_results(
        #         symbol_data_dict[symbol],
        #         model_predict_features_df.loc[test_indices],
        #         y_pred_binary.loc[test_indices],
        #         symbol,
        #     )

    return precision_results
