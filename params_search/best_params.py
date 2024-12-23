import numpy as np


def find_best_params(
    optimal_params, symbol_data_dict, symbol_signals, trading_strategy
):
    best_params = None
    max_profit_loss = -np.inf  # 損益の初期値を最小値に設定
    param_results = []

    print(f"ベストな最適パラメータの探索開始")
    for params in optimal_params:  # パラメータを抽出
        sum_profit_loss = 0  # 各パラメータごとに総損益を計算
        pluses = 0  # 勝率計算用
        minuses = 0  # 勝率計算用
        for symbol, daily_data in symbol_data_dict.items():
            for signal_date in symbol_signals[
                symbol
            ]:  # daily_dataに紐づいたsymbol毎のsignalを抽出
                _, _, _, _, profit_loss = trading_strategy(
                    daily_data.copy(),
                    signal_date,
                    params["stop_loss_percentage"],
                    params["trailing_stop_trigger"],
                    params["trailing_stop_update"],
                )
                sum_profit_loss += profit_loss
                if profit_loss > 0:
                    pluses += 1
                elif profit_loss < 0:
                    minuses += 1

        # 勝率を計算
        win_rate = pluses / (pluses + minuses) if (pluses + minuses) > 0 else 0

        # パラメータごとの結果を保存
        param_results.append(
            {
                "params": params,
                "sum_profit_loss": sum_profit_loss,
                "win_rate": win_rate,
                "pluses": pluses,
                "minuses": minuses,
            }
        )

        # 損益が現在の最大値を上回る場合、パラメータを更新
        if sum_profit_loss > max_profit_loss:
            max_profit_loss = sum_profit_loss
            best_params = params

    return best_params, max_profit_loss, param_results
