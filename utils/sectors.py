from utils.sector_symbols import (
    get_automotive_symbols,
    get_technology_symbols,
    get_financial_symbols,
    get_pharmaceutical_symbols,
    get_food_symbols,
)
from utils.read_symbols_from_csv import read_symbols_from_csv


def get_symbols_by_sector(sector_number):
    # 文字列の場合は整数に変換
    if isinstance(sector_number, str):
        if sector_number.isdigit():
            sector_number = int(sector_number)
        elif sector_number in ["all", "csv", "test"]:
            pass  # "all", "csv", または "test" はそのまま処理
        else:
            raise ValueError(
                "Invalid sector number: must be an integer, 'all', 'csv', or 'test'"
            )

    sector_names = {
        1: "Automotive",
        2: "Technology",
        3: "Financial",
        4: "Pharmaceutical",
        5: "Food",
        "test": "Test",
    }

    switcher = {
        1: get_automotive_symbols,
        2: get_technology_symbols,
        3: get_financial_symbols,
        4: get_pharmaceutical_symbols,
        5: get_food_symbols,
        "test": get_test_symbols,
    }

    if sector_number == "all":
        symbols = get_all_symbols()
        sector_name = "All Sectors"
    elif sector_number == "csv":
        symbols = read_symbols_from_csv("utils/csv/ticker_codes.csv")
        sector_name = "CSV Import"
    elif sector_number == "test":
        symbols = get_test_symbols()
        sector_name = "Test"
    else:
        func = switcher.get(sector_number, None)
        if func is None:
            raise ValueError("Invalid sector number")
        symbols = func()
        # セクター名を取得
        sector_name = sector_names.get(sector_number, "Unknown")

    # 選択されたセクターの銘柄リストを表示
    print(f"Selected sector ({sector_number} - {sector_name})")
    print(f"symbols: {symbols}")

    return symbols


# 一括で返すメソッドを追加
def get_all_symbols():
    return (
        get_automotive_symbols()
        + get_technology_symbols()
        + get_financial_symbols()
        + get_pharmaceutical_symbols()
        + get_food_symbols()
    )


def get_test_symbols():
    return ["7203.T", "8306.T"]
