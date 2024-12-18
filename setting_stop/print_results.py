from setting_stop.trading_strategy import trading_strategy

def print_results(data, trade_start_date, best_result, worst_result):
    # 最適なパラメータの結果を表示
    print("\n最適なパラメータ:")
    print(f"Stop Loss Percentage: {best_result['stop_loss_percentage']}%")
    print(f"Trailing Stop Trigger: {best_result['trailing_stop_trigger']}%")
    print(f"Trailing Stop Update: {best_result['trailing_stop_update']}%")
    print(f"Profit/Loss: {best_result['profit_loss']}%")

    # 最適なパラメータでトレーディングストラテジーを再実行
    purchase_date, purchase_price, end_date, end_price, profit_loss = trading_strategy(
        data.copy(),
        trade_start_date,
        best_result["stop_loss_percentage"],
        best_result["trailing_stop_trigger"],
        best_result["trailing_stop_update"],
    )

    # 保持期間を計算
    holding_period = (end_date - purchase_date).days

    # 出力の修正
    print(f"\n開始日: {purchase_date.date()}, 購入金額: {purchase_price}")
    print(f"終了日: {end_date.date()}, 終了金額: {end_price}")
    print(f"保持期間: {holding_period} 日")
    result = "勝" if profit_loss >= 10 else "負" if profit_loss < 0 else "いずれでもない"
    print(f"損益%: {profit_loss:.2f}%, 結果: {result}")

    # 最悪なパラメータの結果を表示
    print("\n最悪なパラメータ:")
    print(f"Stop Loss Percentage: {worst_result['stop_loss_percentage']}%")
    print(f"Trailing Stop Trigger: {worst_result['trailing_stop_trigger']}%")
    print(f"Trailing Stop Update: {worst_result['trailing_stop_update']}%")
    print(f"Profit/Loss: {worst_result['profit_loss']}%")

    # 最悪なパラメータでトレーディングストラテジーを再実行
    purchase_date, purchase_price, end_date, end_price, profit_loss = trading_strategy(
        data.copy(),
        trade_start_date,
        worst_result["stop_loss_percentage"],
        worst_result["trailing_stop_trigger"],
        worst_result["trailing_stop_update"],
    )

    # 保持期間を計算
    holding_period = (end_date - purchase_date).days

    # 出力の修正
    print(f"\n開始日: {purchase_date.date()}, 購入金額: {purchase_price}")
    print(f"終了日: {end_date.date()}, 終了金額: {end_price}")
    print(f"保持期間: {holding_period} 日")
    result = "勝" if profit_loss >= 10 else "負" if profit_loss < 0 else "いずれでもない"
    print(f"損益%: {profit_loss:.2f}%, 結果: {result}")
