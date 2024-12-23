import pandas as pd
import time
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from features.create_features import create_features
from sectors import get_symbols_by_sector


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
