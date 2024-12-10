# fundamentals_data\fetch_fundamentals.py
# 銘柄の取得期間に関係なく、入手可能な最新の情報を得る

import sys
import os

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import yfinance as yf
from data_processing.fetch_stock_data import fetch_stock_data


def fetch_fundamentals(symbol):
    # ティッカーシンボルに基づいて Yahoo Finance からデータを取得
    ticker = yf.Ticker(symbol)

    # 財務情報を取得
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    cashflow = ticker.cashflow

    return financials, balance_sheet, cashflow


def display_fundamentals(symbol, start_date, end_date):
    # 株価データの取得
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    print(f"Stock Data for {symbol} from {start_date} to {end_date}:")
    print(stock_data.head())

    # ファンダメンタルズ情報の取得
    financials, balance_sheet, cashflow = fetch_fundamentals(symbol)
    print(f"\nFinancials for {symbol}:")
    print(financials)
    print(f"\nBalance Sheet for {symbol}:")
    print(balance_sheet)
    print(f"\nCashflow for {symbol}:")
    print(cashflow)


if __name__ == "__main__":
    symbol = "7203.T"
    start_date = "2023-01-01"  # ダミー
    end_date = "2023-01-01"  # ダミー
    display_fundamentals(symbol, start_date, end_date)
