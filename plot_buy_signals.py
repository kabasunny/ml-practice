import matplotlib.pyplot as plt

def plot_buy_signals(daily_data, features_df, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(daily_data.index, daily_data['Close'], label='Close Price')
    buy_signals = features_df[features_df['Label'] == 1]  # 正解ラベルにマークを付ける
    plt.scatter(buy_signals.index, daily_data.loc[buy_signals.index]['Close'], color='r', label='Buy Signal', alpha=1)
    plt.title(f'{symbol} Stock Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()