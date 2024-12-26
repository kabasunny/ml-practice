def print_ml_stock_response_summary(ml_stock_response, num_items=5):
    print("MLStockResponse (summary):")
    symbol_data = ml_stock_response.symbol_data

    for i, symbol_data_item in enumerate(symbol_data):
        if i < num_items or i >= len(symbol_data) - num_items:
            print(f"Symbol: {symbol_data_item.symbol}")
            print("  Daily Data:")
            for daily_data_item in symbol_data_item.daily_data[:2]:  # 最初の2項目を表示
                print(
                    f"    Date: {daily_data_item.date}, Open: {daily_data_item.open}, Close: {daily_data_item.close}"
                )
            if len(symbol_data_item.daily_data) > num_items * 2:
                print("    ...")
            for daily_data_item in symbol_data_item.daily_data[
                -2:
            ]:  # 最後の2項目を表示
                print(
                    f"    Date: {daily_data_item.date}, Open: {daily_data_item.open}, Close: {daily_data_item.close}"
                )
            print(
                "  Signals:",
                symbol_data_item.signals[:2],
                "...",
                symbol_data_item.signals[-2:],
            )
        if i == num_items - 1 and len(symbol_data) > num_items * 2:
            print("...")
            i = len(symbol_data) - num_items - 1  # 残りの項目をスキップ
