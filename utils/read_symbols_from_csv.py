import csv


def read_symbols_from_csv(file_path):
    symbols = []
    with open(file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # ヘッダーをスキップする場合
        for row in reader:
            ticker = row[0]
            if not ticker.endswith(".T"):  # 末尾が ".T" でない場合
                ticker += ".T"
            symbols.append(ticker)  # ティッカーコードをリストに追加
    return symbols


# テスト用の例
# csv_file_path = "path/to/your/csvfile.csv"
# symbols = read_symbols_from_csv(csv_file_path)
# print(symbols)
