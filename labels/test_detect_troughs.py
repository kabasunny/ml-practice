import numpy as np
from detect_troughs import detect_troughs

# テスト用のサンプルデータ
sample_prices = np.array([10, 7, 5, 6, 8, 4, 3, 4, 8, 6, 2, 5, 3, 4, 6, 7, 5, 2, 3])

# detect_troughs 関数をテスト
troughs, avg_cycle, median_cycle, error_trough, mode_cycle = detect_troughs(sample_prices)

# 結果を出力
print("Detected troughs at indices:", troughs)
print("Average trough cycle:", avg_cycle)
print("Median trough cycle:", median_cycle)
print("Mean absolute error trough:", error_trough)
print("Mode trough cycle:", mode_cycle)

