import pandas as pd
import time
import json
import os
from data_processing.fetch_stock_data import fetch_stock_data
from labels.create_labels import create_labels
from features.create_features import create_features
from utils.sectors import get_symbols_by_sector


def fetch_and_prepare_data(
    sector_number, trade_start_date, before_period_days, end_date, data_numbers
):
    start_time_features = time.time()

    symbols = get_symbols_by_sector(sector_number)
    all_features_df_for_train = pd.DataFrame()
    all_features_df_for_evaluate = pd.DataFrame()
    symbol_data_dict = {}

    if os.path.exists("output_data") and os.listdir("output_data"):
        user_input = input("output_data に既存のファイルがあります。続行しますか？ (Y/N): ")
        if user_input.lower() != 'y':
            print("保存をスキップしました。")
            return load_data("output_data")

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

    end_time_features = time.time()
    print(
        f"データ取得、学習データ、特徴量、ラベルの生成 処理時間: {end_time_features - start_time_features:.2f}秒"
    )

    # データを保存
    save_data(all_features_df_for_train, all_features_df_for_evaluate, symbol_data_dict)

    return all_features_df_for_train, all_features_df_for_evaluate, symbol_data_dict


def save_data(train_df, evaluate_df, symbol_dict, output_dir="output_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df.to_csv(os.path.join(output_dir, "features_df_for_train.csv"), index=False)
    evaluate_df.to_csv(os.path.join(output_dir, "features_df_for_evaluate.csv"), index=False)
    
    # symbol_data_dict内のDate列を文字列に変換
    symbol_dict_serializable = {}
    for symbol, data in symbol_dict.items():
        data = data.reset_index()
        if 'Date' in data.columns:
            data['Date'] = data['Date'].astype(str)
        symbol_dict_serializable[symbol] = data.to_dict(orient="records")
        
    with open(os.path.join(output_dir, "symbol_data_dict.json"), "w") as f:
        json.dump(symbol_dict_serializable, f)

def load_data(input_dir="output_data"):
    train_df = pd.read_csv(os.path.join(input_dir, "features_df_for_train.csv"))
    evaluate_df = pd.read_csv(os.path.join(input_dir, "features_df_for_evaluate.csv"))
    
    with open(os.path.join(input_dir, "symbol_data_dict.json"), "r") as f:
        symbol_dict = {}
        symbol_data = json.load(f)
        for symbol, data in symbol_data.items():
            df = pd.DataFrame(data)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])  # Date列をTimestamp型に変換
                df.set_index('Date', inplace=True)
            symbol_dict[symbol] = df
    
    return train_df, evaluate_df, symbol_dict


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
