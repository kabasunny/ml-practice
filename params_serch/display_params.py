# display_params.py


def display_params(best_params, max_profit_loss, param_results, description):
    # パラメータの表示
    # print(f"総損益: {max_profit_loss:.2f}%")
    # if best_params is not None and not best_params.empty:
    #     print(f"ストップオーダーのパラメータ:")
    #     print(f"初期LC値: {best_params['stop_loss_percentage']}%")
    #     print(f"TSトリガー値: {best_params['trailing_stop_trigger']}%")
    #     print(f"TS更新値: {best_params['trailing_stop_update']}%")
    # else:
    #     print(f"適切なパラメータが見つかりませんでした。")

    # ベスト3とワースト3のパラメータを表示
    sorted_params = sorted(
        param_results, key=lambda x: x["sum_profit_loss"], reverse=True
    )

    print(f"\n{description.upper()} BEST3のパラメータ:")
    for i in range(min(3, len(sorted_params))):
        params = sorted_params[i]["params"]
        print(
            f"初期LC値: {params['stop_loss_percentage']}%, TSトリガー値: {params['trailing_stop_trigger']}%, TS更新値: {params['trailing_stop_update']}%, 総損益: {sorted_params[i]['sum_profit_loss']:.2f}%, 勝率: {sorted_params[i]['win_rate']:.2f}"
        )

    print(f"\n{description.upper()} WORST3のパラメータ:")
    for i in range(min(3, len(sorted_params))):
        params = sorted_params[-(i + 1)]["params"]
        print(
            f"初期LC値: {params['stop_loss_percentage']}%, TSトリガー値: {params['trailing_stop_trigger']}%, TS更新値: {params['trailing_stop_update']}%, 総損益: {sorted_params[-(i+1)]['sum_profit_loss']:.2f}%, 勝率: {sorted_params[-(i+1)]['win_rate']:.2f}"
        )
