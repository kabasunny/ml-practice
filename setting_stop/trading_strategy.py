import pandas as pd

# ロスカットやトレーリングストップの条件を満たすまでデータを評価
# 条件を満たした場合には取引を終了し、終了日の価格、損益、結果を返す

def trading_strategy(data, start_date, stop_loss_percentage, trailing_stop_trigger, trailing_stop_update):
    # データが存在する最初の日付を取得
    while start_date not in data.index:
        start_date += pd.Timedelta(days=1)
    
    initial_price = data.loc[start_date, 'Open']
    stop_loss_threshold = initial_price * stop_loss_percentage / 100
    trailing_stop_price = initial_price * (1 + trailing_stop_update / 100)

    purchase_date = start_date
    purchase_price = initial_price
    end_price = None
    end_date = None
    trade_result = 0

    for current_date in data.index:
        if current_date < start_date:
            continue

        low_price = data.loc[current_date, 'Low']
        open_price = data.loc[current_date, 'Open']
        close_price = data.loc[current_date, 'Close']

        # ロスカット判定
        if purchase_price - low_price >= stop_loss_threshold:
            end_price = low_price
            end_date = current_date
            trade_result = 0
            break
        elif current_date > start_date and (purchase_price - min(open_price, low_price)) >= stop_loss_threshold:
            end_price = min(open_price, low_price)
            end_date = current_date
            trade_result = 0
            break

        # トレーリングストップの更新
        if close_price - purchase_price >= purchase_price * trailing_stop_trigger / 100:
            stop_loss_threshold = trailing_stop_price - purchase_price + purchase_price * (trailing_stop_update / 100)
            trailing_stop_price = close_price * (1 + trailing_stop_update / 100)

    if end_date is None:
        end_price = data.iloc[-1]['Close']
        end_date = data.index[-1]
        trade_result = 1 if end_price > purchase_price else 0

    profit_loss = (end_price - purchase_price) / purchase_price * 100
    return purchase_date, purchase_price, end_date, end_price, profit_loss, trade_result
