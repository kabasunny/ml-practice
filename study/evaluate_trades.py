def evaluate_trades(data, buy_signals, buy_threshold=0.1, sell_threshold=0.05):
    final_signals = []
    purchase_price = None
    trailing_stop_price = None
    profit_trades = 0
    trade_count = 0
    buy_indices = []
    sell_indices = []

    for i in range(len(buy_signals)):
        if buy_signals[i] == 'Buy' and purchase_price is None:
            purchase_price = data['Open'].iloc[i]  # 購入価格を設定
            trailing_stop_price = purchase_price * (1 - sell_threshold)
            trade_count += 1
            buy_indices.append(i)
            print(f"i:{i}, purchase_price:{purchase_price}, trailing_stop_price:{trailing_stop_price}, trade_count:{trade_count}")

        elif purchase_price is not None:
            current_price = data['Close'].iloc[i]
            
            # トレーリングストップの更新
            if current_price > purchase_price * (1 + buy_threshold):
                trailing_stop_price = current_price * (1 - sell_threshold)
                purchase_price = current_price
                print(f"i:{i}, updated purchase_price:{purchase_price}, trailing_stop_price:{trailing_stop_price}")
            
            # ロスカットまたはトレーリングストップで売却
            if current_price < trailing_stop_price:
                final_signals.append('Sell')
                if current_price > purchase_price:
                    profit_trades += 1
                sell_indices.append(i)
                print(f"i:{i}, Sell at current_price:{current_price}, profit_trades:{profit_trades}")
                purchase_price = None
                trailing_stop_price = None
            else:
                final_signals.append('Hold')
        else:
            final_signals.append('Hold')

    # 結果の評価
    success_ratio = profit_trades / trade_count if trade_count > 0 else 0
    print(f'Success Ratio: {success_ratio * 100:.2f}%')
    return final_signals, buy_indices, sell_indices
