import pandas as pd


# ロスカットやトレーリングストップの条件を満たすまでデータを評価
# 条件を満たした場合には取引を終了し、終了日の価格、損益、結果を返す
def trading_strategy(
    data, start_date, stop_loss_percentage, trailing_stop_trigger, trailing_stop_update
):
    # データが存在する最初の日付を取得
    while start_date not in data.index:
        start_date += pd.Timedelta(days=1)  # データに存在する日付になるまで日付を進める

    initial_price = data.loc[start_date, "Open"]  # 取引開始日の始値を取得
    stop_loss_threshold = (
        initial_price * stop_loss_percentage / 100
    )  # ロスカット閾値を計算
    trailing_stop_price = initial_price * (
        1 + trailing_stop_update / 100
    )  # トレーリングストップの初期価格を計算

    purchase_date = start_date  # 取引開始日を設定
    purchase_price = initial_price  # 取引開始価格を設定
    end_price = None  # 取引終了価格を初期化
    end_date = None  # 取引終了日を初期化
    trade_result = 0  # 取引結果を初期化（0: 損失, 1: 利益）

    for current_date in data.index:
        if current_date < start_date:
            continue  # 取引開始日以前の日付はスキップ

        low_price = data.loc[current_date, "Low"]  # 当日の安値を取得
        open_price = data.loc[current_date, "Open"]  # 当日の始値を取得
        close_price = data.loc[current_date, "Close"]  # 当日の終値を取得

        # ロスカット判定
        if purchase_price - low_price >= stop_loss_threshold:
            end_price = low_price  # ロスカット価格を設定
            end_date = current_date  # 取引終了日を設定
            trade_result = 0  # 取引結果を損失とする
            break
        elif (
            current_date > start_date
            and (purchase_price - min(open_price, low_price)) >= stop_loss_threshold
        ):
            end_price = min(open_price, low_price)  # ロスカット価格を設定
            end_date = current_date  # 取引終了日を設定
            trade_result = 0  # 取引結果を損失とする
            break

        # トレーリングストップの更新
        if close_price - purchase_price >= purchase_price * trailing_stop_trigger / 100:
            stop_loss_threshold = (
                trailing_stop_price
                - purchase_price
                + purchase_price * (trailing_stop_update / 100)
            )
            trailing_stop_price = close_price * (1 + trailing_stop_update / 100)

    # 取引終了条件が満たされなかった場合
    if end_date is None:
        end_price = data.iloc[-1]["Close"]  # 最終日の終値を設定
        end_date = data.index[-1]  # 最終日を取引終了日とする
        trade_result = 1 if end_price > purchase_price else 0  # 利益が出ているかを判定

    profit_loss = (
        (end_price - purchase_price) / purchase_price * 100
    )  # 損益を計算（パーセンテージ）
    return (
        purchase_date,
        purchase_price,
        end_date,
        end_price,
        profit_loss,
        trade_result,
    )  # 結果を返す
