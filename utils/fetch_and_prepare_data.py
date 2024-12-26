import pandas as pd
import time
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from features.create_features import create_features
from utils.sectors import get_symbols_by_sector


def fetch_and_prepare_data(
    sector_number, trade_start_date, before_period_days, end_date, data_numbers
):
    symbols = get_symbols_by_sector(sector_number)
    all_features_df_for_train = pd.DataFrame()
    all_features_df_for_evaluate = pd.DataFrame()
    symbol_data_dict = {}

    for symbol in symbols:
        try:
            start_date = trade_start_date - pd.Timedelta(days=before_period_days)
            daily_data = fetch_stock_data(symbol, start_date, end_date)
            if daily_data.empty:
                print(f"データが見つかりませんでした: {symbol}")
                continue
            labeled_data = create_labels(daily_data)
            features_df_for_train, features_all_df_evaluate = create_features(
                daily_data, trade_start_date, labeled_data, data_numbers
            )
            features_df_for_train["Symbol"] = symbol
            features_all_df_evaluate["Symbol"] = symbol
            features_df_for_train.dropna(inplace=True)
            features_all_df_evaluate.dropna(inplace=True)
            all_features_df_for_train = pd.concat(
                [all_features_df_for_train, features_df_for_train]
            )
            all_features_df_for_evaluate = pd.concat(
                [all_features_df_for_evaluate, features_all_df_evaluate]
            )
            symbol_data_dict[symbol] = daily_data
            time.sleep(1)
        except Exception as e:
            print(f"エラーが発生しました: {symbol}, {e}")

    return all_features_df_for_train, all_features_df_for_evaluate, symbol_data_dict


# symbol_data_dictの例
# {
#     'AAPL':
#           Open    High     Low   Close    Adj Close    Volume
# Date
# 2023-12-01  2819.0  2842.0  2803.0  2833.0  2758.835693  26774000
# 2023-12-04  2802.0  2802.5  2744.5  2767.5  2695.050293  30495700
# 2023-12-05  2770.0  2784.5  2743.5  2753.5  2681.416748  24512600,

#     'GOOGL':
#           Open    High     Low   Close    Adj Close    Volume
# Date
# 2023-12-01  1400.0  1412.0  1393.0  1405.0  1380.56789  12774000
# 2023-12-04  1392.0  1394.5  1375.5  1387.5  1365.12345  13000000
# 2023-12-05  1380.0  1386.5  1368.5  1373.5  1350.87654  12000000
# }
