import pandas as pd
from fetch_stock_data import fetch_stock_data
from preprocess_data import preprocess_data
from random_daily_buy import random_daily_buy
from evaluate_trades import evaluate_trades
from plot_trades import plot_trades

# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
start_date = "2023-09-01"
end_date = "2023-12-31"

data = fetch_stock_data(symbol, start_date, end_date)
data = preprocess_data(data)
buy_signals = random_daily_buy(data)
final_signals, buy_indices, sell_indices = evaluate_trades(data, buy_signals)
print(final_signals)

# 買いと売りのタイミングをチャートにプロット
plot_trades(data, buy_indices, sell_indices)
