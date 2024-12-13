# setting_stop\main.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 日本語フォントを設定
font_path = "C:/Windows/Fonts/msgothic.ttc"  # ゴシック体のフォントパスを指定
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# あなたのモジュールをインポート
from data_processing.fetch_stock_data import fetch_stock_data
from setting_stop.strategies.trading_strategy import trading_strategy
from setting_stop.plot_stop_results import plot_stop_results

# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
start_date = "2022-01-01"
end_date = "2023-12-31"
trade_start_date = pd.Timestamp("2022-03-08")  # 買いを入れる日

data = fetch_stock_data(symbol, start_date, end_date)

print(f"銘柄コード: {symbol} , チャート期間: {start_date} 〜 {end_date}")

# パラメータ最適化の範囲を設定
stop_loss_percentages = np.arange(1, 11, 1)  # 1%から10%まで、1%刻み
trailing_stop_triggers = np.arange(1, 11, 1)  # 1%から10%まで、1%刻み
trailing_stop_updates = np.arange(0.5, 5.5, 0.5)  # 0.5%から5%まで、0.5%刻み

results = []

# パラメータの全組み合わせを生成
parameter_combinations = list(
    product(stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates)
)

# グリッドサーチを実行
print("パラメータ最適化中...")
for stop_loss_percentage, trailing_stop_trigger, trailing_stop_update in tqdm(
    parameter_combinations
):
    try:
        # トレーディングストラテジーを実行
        purchase_date, purchase_price, exit_date, exit_price, profit_loss = (
            trading_strategy(
                data.copy(),
                trade_start_date,
                stop_loss_percentage,
                trailing_stop_trigger,
                trailing_stop_update,
            )
        )
        results.append(
            {
                "stop_loss_percentage": stop_loss_percentage,
                "trailing_stop_trigger": trailing_stop_trigger,
                "trailing_stop_update": trailing_stop_update,
                "profit_loss": profit_loss,
                "purchase_date": purchase_date,
                "exit_date": exit_date,
            }
        )
    except Exception as e:
        # 必要に応じてエラーをログに記録
        # print(f"エラー: {e}")
        continue

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 損益でソートして最適なパラメータを見つける
sorted_results = results_df.sort_values(by="profit_loss", ascending=False)
best_result = sorted_results.iloc[0]

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

print(f"\n開始日: {purchase_date.date()}, 購入金額: {purchase_price}")
print(f"終了日: {end_date.date()}, 終了金額: {end_price}")
result = "勝" if profit_loss >= 10 else "いずれでもない" if profit_loss > -10 else "負"
print(f"損益%: {profit_loss:.2f}%, 結果: {result}")

# 結果のプロット
data["Date"] = data.index  # Date列を追加
plot_stop_results(data, purchase_date, purchase_price, end_date, end_price)

# 結果をCSVファイルに保存
results_df.to_csv("optimization_results.csv", index=False)
best_result.to_csv("best_params.csv", index=False)

# 最適化の結果をヒートマップでプロット
fixed_trigger = best_result[
    "trailing_stop_trigger"
]  # 最適なトレーリングストップトリガーを固定
subset = results_df[results_df["trailing_stop_trigger"] == fixed_trigger]

# ピボットテーブルを作成
pivot_table = subset.pivot_table(
    index="stop_loss_percentage",
    columns="trailing_stop_update",
    values="profit_loss",
    aggfunc="mean",
)

# ヒートマップを描画
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm")
plt.title(f"損益ヒートマップ (Trailing Stop Trigger = {fixed_trigger}%)")
plt.xlabel("Trailing Stop Update (%)")
plt.ylabel("Stop Loss Percentage (%)")
plt.show()
