import matplotlib.pyplot as plt
import pandas as pd  # pandasをインポート

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

def plot_results(daily_data, features_df, results_df, symbol):
    plt.figure(figsize=(14, 7))
    plt.plot(daily_data.index, daily_data['Close'], label='Close Price')
    
    actual_buy_signals = features_df[features_df['Label'] == 1].index
    actual_not_buy_signals = features_df[features_df['Label'] == 0].index
    predicted_buy_signals = results_df[(results_df['Predicted'] == 1) & (results_df['Symbol'] == symbol)].index
    
    correct_predictions = results_df[(results_df['Actual'] == 1) & (results_df['Predicted'] == 1) & (results_df['Symbol'] == symbol)].index
    incorrect_predictions = results_df[(results_df['Actual'] == 0) & (results_df['Predicted'] == 1) & (results_df['Symbol'] == symbol)].index

    plt.scatter(actual_buy_signals, daily_data.loc[actual_buy_signals]['Close'], marker='^', color='g', label='Actual Buy Signal', s=100)
    plt.scatter(actual_not_buy_signals, daily_data.loc[actual_not_buy_signals]['Close'], marker='v', color='orange', label='Actual Not Buy Signal', s=50)
    plt.scatter(correct_predictions, daily_data.loc[correct_predictions]['Close'], marker='o', color='b', label='Correct Prediction', s=100)
    plt.scatter(incorrect_predictions, daily_data.loc[incorrect_predictions]['Close'], marker='x', color='r', label='Incorrect Prediction', s=100)
    
    plt.title(f'{symbol} - Actual vs Predicted Buy Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()
