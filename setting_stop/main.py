import sys
import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# モジュールをインポート
from data_processing.fetch_stock_data import fetch_stock_data
from setting_stop.trading_strategy import trading_strategy
from setting_stop.plot_stop_results import plot_stop_results
from setting_stop.optimize_parameters import optimize_parameters
from setting_stop.plot_heatmap import plot_heatmap
from print_results import print_results  # 新しいファイルをインポート

# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
trade_start_date = pd.Timestamp("2019-01-01")  # 買いを入れる日
period_days = 365 * 2  # 前後2年を期間とする例

# start_date と end_date を trade_start_date を基に設定
start_date = trade_start_date - pd.Timedelta(days=period_days)
end_date = trade_start_date + pd.Timedelta(days=period_days)

data = fetch_stock_data(symbol, start_date, end_date)

print(f"銘柄コード: {symbol} , チャート期間: {start_date.date()} 〜 {end_date.date()}")

# パラメータ最適化の実行
best_result, worst_result, results_df = optimize_parameters(data, trade_start_date)

# ベスト結果のトレーディングストラテジーの再実行
best_purchase_date, best_purchase_price, best_end_date, best_end_price, _ = (
    trading_strategy(
        data.copy(),
        trade_start_date,
        best_result["stop_loss_percentage"],
        best_result["trailing_stop_trigger"],
        best_result["trailing_stop_update"],
    )
)

# ワースト結果のトレーディングストラテジーの再実行
worst_purchase_date, worst_purchase_price, worst_end_date, worst_end_price, _ = (
    trading_strategy(
        data.copy(),
        trade_start_date,
        worst_result["stop_loss_percentage"],
        worst_result["trailing_stop_trigger"],
        worst_result["trailing_stop_update"],
    )
)

# 結果の表示
print_results(data, trade_start_date, best_result, worst_result)

# ベスト結果のプロット
data["Date"] = data.index  # Date列を追加
plot_stop_results(
    "BEST", data, best_purchase_date, best_purchase_price, best_end_date, best_end_price
)

# ワースト結果のプロット
plot_stop_results(
    "WORST",
    data,
    worst_purchase_date,
    worst_purchase_price,
    worst_end_date,
    worst_end_price,
)

# # 結果をCSVファイルに保存
# results_df.to_csv("optimization_results.csv", index=False)
# best_result.to_csv("best_params.csv", index=False)

# 最適化の結果をヒートマップでプロット
fixed_trigger = best_result[
    "trailing_stop_trigger"
]  # 最適なトレーリングストップトリガーを固定
subset = results_df[results_df["trailing_stop_trigger"] == fixed_trigger]

# ヒートマップをプロット
plot_heatmap(subset, fixed_trigger)

# 最適化の結果をヒートマップでプロット
fixed_trigger = worst_result[
    "trailing_stop_trigger"
]  # 最適なトレーリングストップトリガーを固定
subset = results_df[results_df["trailing_stop_trigger"] == fixed_trigger]

# ヒートマップをプロット
plot_heatmap(subset, fixed_trigger)
