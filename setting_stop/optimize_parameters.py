import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from setting_stop.trading_strategy import trading_strategy


def optimize_parameters(data, trade_start_date):
    results = []

    # パラメータ最適化の範囲を設定
    stop_loss_percentages = np.arange(2, 4, 1)  # エントリー時のロスカット
    trailing_stop_triggers = np.arange(5, 10, 1)  # TSを引き上げる閾値
    trailing_stop_updates = np.arange(2, 4, 1)  # 現在価格に対するストップ

    # パラメータの全組み合わせを生成
    parameter_combinations = list(
        product(stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates)
    )

    # グリッドサーチを実行
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

    # 損益でソートして最適なパラメータと最悪なパラメータを見つける
    # 最良の結果が複数ある場合、stop_loss_percentages, trailing_stop_triggers, trailing_stop_updatesの値がそれぞれ最も小さいケースを選ぶ
    sorted_results = results_df.sort_values(
        by=[
            "profit_loss",
            "stop_loss_percentage",
            "trailing_stop_trigger",
            "trailing_stop_update",
        ],
        ascending=[False, True, True, True],
    )
    # ascending パラメータには、各列に対して昇順（True）または降順（False）でソートするかを指定
    best_result = sorted_results.iloc[0]
    worst_result = sorted_results.iloc[-1]

    return best_result, worst_result, results_df
