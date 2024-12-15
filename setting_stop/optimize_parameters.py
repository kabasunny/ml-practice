import pandas as pd
from itertools import product
from tqdm import tqdm
from setting_stop.trading_strategy import trading_strategy

def optimize_parameters(data, trade_start_date, stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates):
    results = []

    # パラメータの全組み合わせを生成
    parameter_combinations = list(
        product(stop_loss_percentages, trailing_stop_triggers, trailing_stop_updates)
    )

    # グリッドサーチを実行
    print("パラメータ最適化中...")
    for stop_loss_percentage, trailing_stop_trigger, trailing_stop_update in tqdm(
        parameter_combinations
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
