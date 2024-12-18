# setting_stop/plot_heatmap.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(subset, fixed_trigger):
    # ピボットテーブルを作成
    pivot_table = subset.pivot_table(
        index="stop_loss_percentage",
        columns="trailing_stop_update",
        values="profit_loss",
        aggfunc="mean",
    )

    # ヒートマップを描画
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(f"Profit and Loss Heatma (Trailing Stop Trigger = {fixed_trigger}%)")
    plt.xlabel("Trailing Stop Update (%)")
    plt.ylabel("Stop Loss Percentage (%)")
    plt.show()
