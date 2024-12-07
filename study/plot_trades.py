import matplotlib.pyplot as plt

def plot_trades(data, buy_indices, sell_indices):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', marker='o')
    
    # 高さを調整するためのオフセット
    y_offset = max(data['Close']) * 0.05
    
    for i in buy_indices:
        plt.text(data['Date'].iloc[i], data['Close'].iloc[i] + y_offset, f'B{i}', color='red', fontsize=12, ha='center')
        plt.vlines(data['Date'].iloc[i], data['Close'].iloc[i], data['Close'].iloc[i] + y_offset, color='red', linestyle='dashed')
    
    for i in sell_indices:
        plt.text(data['Date'].iloc[i], data['Close'].iloc[i] + y_offset, f'S{i}', color='blue', fontsize=12, ha='center')
        plt.vlines(data['Date'].iloc[i], data['Close'].iloc[i], data['Close'].iloc[i] + y_offset, color='blue', linestyle='dashed')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Buy and Sell Signals')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
