import os
import json
import pandas as pd


def save_data(train_df, evaluate_df, symbol_dict, output_dir="output_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df.to_csv(os.path.join(output_dir, "features_df_for_train.csv"), index=False)
    evaluate_df.to_csv(
        os.path.join(output_dir, "features_df_for_evaluate.csv"), index=False
    )

    # symbol_data_dict内のDate列を文字列に変換
    symbol_dict_serializable = {}
    for symbol, data in symbol_dict.items():
        data = data.reset_index()
        if "Date" in data.columns:
            data["Date"] = data["Date"].astype(str)
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
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])  # Date列をTimestamp型に変換
                df.set_index("Date", inplace=True)
            symbol_dict[symbol] = df

    return train_df, evaluate_df, symbol_dict
