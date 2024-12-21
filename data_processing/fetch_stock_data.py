# data_processing\fetch_stock_data.py
import yfinance as yf
import pandas as pd

# pandas.DataFrame を戻す
def fetch_stock_data(symbol, start_date, end_date):
    # 日足データの取得
    daily_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    
    return daily_data # pandas.DataFrame を戻す


#     """
#                   Open    High     Low   Close    Adj Close    Volume
#     Date
#     2023-12-01  2819.0  2842.0  2803.0  2833.0  2758.835693  26774000
#     2023-12-04  2802.0  2802.5  2744.5  2767.5  2695.050293  30495700
#     2023-12-05  2770.0  2784.5  2743.5  2753.5  2681.416748  24512600
#     """


