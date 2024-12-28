import csv
import os


# ensemble_methods/evaluate_ensemble.pyのsave_signals_to_csv関数
# Goで売りパラメータのグリッドサーチに突っ込む目的
def save_signals_to_csv(precision_results, output_dir="output_signals"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for result in precision_results:
        combination, metrics, duplicated_values = result
        for symbol, signal_dates in duplicated_values.items():
            file_path = os.path.join(output_dir, f"{symbol}_signals.csv")
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Date"])  # ヘッダー行
                for date in signal_dates:
                    writer.writerow([date])
