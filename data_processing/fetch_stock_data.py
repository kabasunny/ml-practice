# data_processing\fetch_stock_data.py
import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start_date, end_date):
    # 日足データの取得
    daily_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    # # 週足データの取得
    # weekly_data = yf.download(symbol, start=start_date, end=end_date, interval='1wk')
    # # 月足データの取得
    # monthly_data = yf.download(symbol, start=start_date, end=end_date, interval='1mo')

    # リサンプリングで週足データと月足データを生成
    # weekly_data = daily_data.resample("W").ffill()
    # monthly_data = daily_data.resample("ME").ffill()

    # return daily_data, weekly_data, monthly_data, weekly_data_rs, monthly_data_rs
    return daily_data

# def fetch_stock_data(symbol, start_date, end_date):
#     data, _, _ = yf.download(symbol, start=start_date, end=end_date)
#     # print(data.head())
#     """
#                   Open    High     Low   Close    Adj Close    Volume
#     Date
#     2023-12-01  2819.0  2842.0  2803.0  2833.0  2758.835693  26774000
#     2023-12-04  2802.0  2802.5  2744.5  2767.5  2695.050293  30495700
#     2023-12-05  2770.0  2784.5  2743.5  2753.5  2681.416748  24512600
#     """
#     return data



def main():
    symbol = "7203.T"  # トヨタ自動車の例
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    daily_data, weekly_data, _ = fetch_stock_data(symbol, start_date, end_date)

    # Closeが等しい日付の抽出
    matching_close_dates = daily_data[daily_data['Close'].isin(weekly_data['Close'])].index
    print("Closeが等しい日付:")
    print(matching_close_dates)

    # Close, Adj Close, Volumeが等しい日付の抽出
    matching_all_dates = daily_data[
        (daily_data['Close'].isin(weekly_data['Close'])) &
        (daily_data['Adj Close'].isin(weekly_data['Adj Close'])) &
        (daily_data['Volume'].isin(weekly_data['Volume']))
    ].index
    print("\nClose, Adj Close, Volumeが等しい日付:")
    print(matching_all_dates)

    # 日付の差分データの表示
    diff_dates = matching_close_dates.difference(matching_all_dates)
    print("\n差分の日付:")
    print(diff_dates)

    # diff_datesの日付を持ったdaily_dataの株価データを表示 
    diff_data = daily_data.loc[diff_dates] 
    print("\ndiff_datesの日付を持ったdaily_dataの株価データ:")
    print(diff_data)    

    # diff_dataと同じ終値を持ったweekly_dataの株価データを表示
    matching_weekly_data = weekly_data[weekly_data['Close'].isin(diff_data['Close'])]
    print("\ndiff_dataと同じ終値を持ったweekly_dataの株価データ:")
    print(matching_weekly_data)


if __name__ == "__main__":
    main()

