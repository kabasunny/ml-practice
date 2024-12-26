import pandas as pd
from proto_definitions.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
)
from proto_definitions.print_ml_stock_response import print_ml_stock_response_summary


def convert_to_proto_response(symbol_signals, symbol_data_dict):
    symbol_data_list = []

    for symbol, signals in symbol_signals.items():
        daily_data_list = [
            MLDailyData(
                date=row.name.strftime("%Y-%m-%d"),
                open=row["Open"],
                high=row["High"],
                low=row["Low"],
                close=row["Close"],
                adj_close=row["Adj Close"],
                volume=int(row["Volume"]),
            )
            for index, row in symbol_data_dict[symbol].iterrows()
        ]
        symbol_data = MLSymbolData(
            symbol=symbol,
            daily_data=daily_data_list,
            signals=[signal.strftime("%Y-%m-%d") for signal in signals],
        )
        symbol_data_list.append(symbol_data)

    ml_stock_response = MLStockResponse(symbol_data=symbol_data_list)

    return ml_stock_response

    # print_ml_stock_response_summary(ml_stock_response)


# MLStockResponse (summary):
# Symbol: 8306.T
#   Daily Data:
#     Date: 2005-09-29, Open: 1460.0, Close: 1450.0
#     Date: 2005-09-30, Open: 1460.0, Close: 1490.0
#     ...
#     Date: 2024-12-25, Open: 1801.5, Close: 1800.0
#     Date: 2024-12-26, Open: 1800.0, Close: 1800.5
#   Signals: ['2015-01-19', '2011-03-17'] ... ['2024-12-25', '2024-12-26']
