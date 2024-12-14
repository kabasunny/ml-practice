# setting_stop/main.py
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

# 日本語フォントを設定
font_path = "C:/Windows/Fonts/msgothic.ttc"  # ゴシック体のフォントパスを指定
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# モジュールをインポート
from data_processing.fetch_stock_data import fetch_stock_data
from setting_stop.trading_strategy import trading_strategy
from setting_stop.plot_stop_results import plot_stop_results
from setting_stop.optimize_parameters import optimize_parameters
from setting_stop.plot_heatmap import plot_heatmap

# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
trade_start_date = pd.Timestamp("2022-02-01")  # 買いを入れる日
period_days = 365 * 1  # 前後1年を期間とする例

# start_date と end_date を trade_start_date を基に設定
start_date = trade_start_date - pd.Timedelta(days=period_days)
end_date = trade_start_date + pd.Timedelta(days=period_days)

data = fetch_stock_data(symbol, start_date, end_date)

print(f"銘柄コード: {symbol} , チャート期間: {start_date.date()} 〜 {end_date.date()}")

# パラメータ最適化の範囲を設定
stop_loss_percentages = np.arange(1, 10, 1)  # 1%から9%まで、1%刻み
trailing_stop_triggers = np.arange(5, 20, 1)  # 5%から19%まで、1%刻み
trailing_stop_updates = np.arange(2, 10, 0.5)  # 2%から9.5%まで、0.5%刻み

# パラメータ最適化の実行
best_result, results_df = optimize_parameters(data, trade_start_date, stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates)

print("\n最適なパラメータ:")
print(f"Stop Loss Percentage: {best_result['stop_loss_percentage']}%")
print(f"Trailing Stop Trigger: {best_result['trailing_stop_trigger']}%")
print(f"Trailing Stop Update: {best_result['trailing_stop_update']}%")
print(f"Profit/Loss: {best_result['profit_loss']}%")

# 最適なパラメータでトレーディングストラテジーを再実行
purchase_date, purchase_price, end_date, end_price, profit_loss = trading_strategy(
    data.copy(),
    trade_start_date,
    best_result["stop_loss_percentage"],
    best_result["trailing_stop_trigger"],
    best_result["trailing_stop_update"],
)

# 保持期間を計算
holding_period = (end_date - purchase_date).days

# 出力の修正
print(f"\n開始日: {purchase_date.date()}, 購入金額: {purchase_price}")
print(f"終了日: {end_date.date()}, 終了金額: {end_price}")
print(f"保持期間: {holding_period} 日")
result = "勝" if profit_loss >= 10 else "負" if profit_loss < 0 else "いずれでもない"
print(f"損益%: {profit_loss:.2f}%, 結果: {result}")

# # 結果のプロット
# data["Date"] = data.index  # Date列を追加
# plot_stop_results(data, purchase_date, purchase_price, end_date, end_price)

# # 結果をCSVファイルに保存
# results_df.to_csv("optimization_results.csv", index=False)
# best_result.to_csv("best_params.csv", index=False)

# # 最適化の結果をヒートマップでプロット
# fixed_trigger = best_result[
#     "trailing_stop_trigger"
# ]  # 最適なトレーリングストップトリガーを固定
# subset = results_df[results_df["trailing_stop_trigger"] == fixed_trigger]

# # ヒートマップをプロット
# plot_heatmap(subset, fixed_trigger)
