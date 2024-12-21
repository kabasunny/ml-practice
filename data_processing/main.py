# data_processing\main.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices

def main():
    symbol = "7203.T"  # トヨタ自動車の例
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    daily_data = fetch_stock_data(symbol, start_date, end_date)

    # トレンド除去
    detrended_prices_T = detrend_prices(daily_data, remove_trend=True)
    detrended_prices_F = detrend_prices(daily_data, remove_trend=False)

    print("Original Data:")
    print(daily_data)
    print("\nDetrended Prices T:")
    print(detrended_prices_T)
    print("\nDetrended Prices F:")
    print(detrended_prices_F)

if __name__ == "__main__":
    main()
