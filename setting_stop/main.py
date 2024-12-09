# setting_stop\main.py

import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from fetch_stock_data import fetch_stock_data
from trading_strategy import trading_strategy
from setting_stop.plot_stop_results import plot_stop_results

# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
start_date = "2022-01-01"
end_date = "2023-12-31"
# trade_start_date = pd.Timestamp(start_date)  # 買いを入れる日
trade_start_date = pd.Timestamp("2022-03-08")  # 買いを入れる日

# ロスカットとトレーリングストップの設定値
stop_loss_percentage = 3.0  # ロスカット閾値（%）
trailing_stop_trigger = 5.0  # トレーリングストップが更新されるための上昇閾値（%）
trailing_stop_update = 2.0  # トレーリングストップの更新値（%）

data = fetch_stock_data(symbol, start_date, end_date)

purchase_date, purchase_price, end_date, end_price, profit_loss, trade_result = (
    trading_strategy(
        data,
        trade_start_date,
        stop_loss_percentage,
        trailing_stop_trigger,
        trailing_stop_update,
    )
)

print(f"開始日: {purchase_date}")
print(f"購入金額: {purchase_price}")
print(f"終了日: {end_date}")
print(f"終了金額: {end_price}")
print(f"損益%: {profit_loss:.2f}%")
print(f"結果: {'1 : 勝ち' if trade_result == 1 else '0 : 負け'}")

# 結果のプロット
data["Date"] = data.index  # Date列を追加
plot_stop_results(
    data, purchase_date, purchase_price, end_date, end_price
)  # コメントアウト済み
