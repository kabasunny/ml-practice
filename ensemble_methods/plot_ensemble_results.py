import matplotlib.pyplot as plt


def plot_ensemble_results(daily_data, features_df, y_pred_binary, symbol):
    """
    アンサンブル評価の結果をプロットする関数
    :param daily_data: 日次データ
    :param features_df: 特徴データフレーム
    :param y_pred_binary: バイナリ予測
    :param symbol: シンボル
    """
    plt.figure(figsize=(14, 7))
    plt.plot(daily_data.index, daily_data["Close"], label="Close Price")

    # 実際の「買い」シグナルと「買いではない」シグナルを抽出
    actual_buy_signals = features_df[features_df["Label"] == 1].index
    actual_buy_signals = actual_buy_signals.intersection(daily_data.index)

    # 予測された「買い」シグナルのみを抽出
    predicted_buy_signals = y_pred_binary[y_pred_binary == 1].index
    predicted_buy_signals = predicted_buy_signals.intersection(daily_data.index)

    # 実際の「買い」シグナルをプロット
    plt.scatter(
        actual_buy_signals,
        daily_data.loc[actual_buy_signals]["Close"],
        marker="^",
        color="g",
        label="Actual Buy Signal",
        s=100,
    )

    # 予測された「買い」シグナルをプロット
    plt.scatter(
        predicted_buy_signals,
        daily_data.loc[predicted_buy_signals]["Close"],
        marker="o",
        color="b",
        label="Predicted Buy Signal",
        s=100,
    )

    plt.title(f"{symbol} - Actual and Predicted Buy Signals (Ensemble)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()
