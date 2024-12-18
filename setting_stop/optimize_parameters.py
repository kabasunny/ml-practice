import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from setting_stop.trading_strategy import trading_strategy


def optimize_parameters(data, trade_start_date):
    results = []

    # パラメータ最適化の範囲を設定
    stop_loss_percentages = np.arange(2, 5, 1)  # 1%から9%まで、1%刻み
    trailing_stop_triggers = np.arange(5, 8, 1)  # 5%から19%まで、1%刻み
    trailing_stop_updates = np.arange(5, 8, 1)  # 2%から9.5%まで、0.5%刻み

    # パラメータの全組み合わせを生成
    parameter_combinations = list(
        product(stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates)
    )
    # +print(f"len(parameter_combinations) : {len(parameter_combinations)}")

    # グリッドサーチを実行
    # print("パラメータ最適化中...")
    for stop_loss_percentage, trailing_stop_trigger, trailing_stop_update in tqdm(
        parameter_combinations, disable=True
    ):
        try:
            # トレーディングストラテジーを実行
            purchase_date, purchase_price, exit_date, exit_price, profit_loss = (
                trading_strategy(
                    data.copy(),
                    trade_start_date,
                    stop_loss_percentage,
                    trailing_stop_trigger,
                    trailing_stop_update,
                )
            )
            results.append(
                {
                    "stop_loss_percentage": stop_loss_percentage,
                    "trailing_stop_trigger": trailing_stop_trigger,
                    "trailing_stop_update": trailing_stop_update,
                    "profit_loss": profit_loss,
                    "purchase_date": purchase_date,
                    "exit_date": exit_date,
                }
            )
        except Exception as e:
            # 必要に応じてエラーをログに記録
            # print(f"エラー: {e}")
            continue

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # 損益でソートして最適なパラメータを見つける
    sorted_results = results_df.sort_values(by="profit_loss", ascending=False)
    best_result = sorted_results.iloc[0]

    return best_result, results_df
