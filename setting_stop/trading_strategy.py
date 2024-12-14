import pandas as pd


def trading_strategy(
    data, start_date, stop_loss_percentage, trailing_stop_trigger, trailing_stop_update
):
    # データが存在する最初の日付を取得
    while start_date not in data.index:
        start_date += pd.Timedelta(days=1)  # データに存在する日付になるまで日付を進める

    # 購入初日の設定
    purchase_date = start_date
    purchase_price = data.loc[start_date, "Open"]  # 取引開始日の始値
    stop_loss_threshold = round(
        purchase_price * (1 - stop_loss_percentage / 100), 1
    )  # 初期ロスカット値
    trailing_stop_trigger_price = round(
        purchase_price * (1 + trailing_stop_trigger / 100), 1
    )  # 初期トレーリングストップ値

    # print(f"取引開始日: {purchase_date}, 始値: {purchase_price}")
    # print(
    #     f"初期LC値: {round(stop_loss_threshold, 1)}, 初期TS発動値: {round(trailing_stop_trigger_price, 1)}"
    # )

    # 初期化
    end_date = None
    end_price = None

    # 取引日ごとの確認
    for current_date in data.index[data.index.get_loc(start_date) :]:
        open_price = data.loc[current_date, "Open"]
        low_price = data.loc[current_date, "Low"]
        close_price = data.loc[current_date, "Close"]

        # print(
        #     f"日付: {current_date}, 始値: {open_price}, 安値: {low_price}, 終値: {close_price}"
        # )

        # ロスカット条件: 当日の始値がロスカット値以下
        if open_price <= stop_loss_threshold:
            end_price = open_price
            end_date = current_date
            # print(f"LC発動: 日付 {end_date}, 終了価格 {end_price}")
            break

        # トレーリングストップ発動条件: 当日の安値がトレーリングストップ値以下
        if low_price <= stop_loss_threshold:
            end_price = low_price
            end_date = current_date
            # print(f"TS発動: 日付 {end_date}, 終了価格 {end_price}")
            break

        # トレーリングストップ更新条件: 終値がトリガーを超えた場合
        if close_price >= trailing_stop_trigger_price:
            stop_loss_threshold = round(
                close_price * (1 - trailing_stop_update / 100), 1
            )
            trailing_stop_trigger_price = round(
                close_price * (1 + trailing_stop_trigger / 100), 1
            )
            # print(
            #     f"SL更新: 日付 {current_date}, SL値(新LC値) {stop_loss_threshold}, 次TS発動値 {trailing_stop_trigger_price}"
            # )

    # 取引終了条件が満たされなかった場合
    if end_date is None:
        end_price = data.iloc[-1]["Close"]  # 最終日の終値
        end_date = data.index[-1]  # 最終日

    # 損益の計算
    profit_loss = round((end_price - purchase_price) / purchase_price * 100, 1)

    # 結果の返却
    return (
        purchase_date,
        purchase_price,
        end_date,
        end_price,
        profit_loss,
    )
