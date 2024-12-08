import matplotlib.pyplot as plt

def plot_results(data, purchase_date, purchase_price, end_date, end_price):
    plt.figure(figsize=(14, 7))
    
    # 株価の推移をプロット
    plt.plot(data['Date'], data['Close'], label='Close Price', marker='o')
    
    # 購入点をプロット
    plt.scatter(purchase_date, purchase_price, color='green', label='Purchase', zorder=5)
    plt.text(purchase_date, purchase_price, f' Purchase\n{purchase_price}', color='green', fontsize=12, ha='left')
    
    # 売却点をプロット
    plt.scatter(end_date, end_price, color='red', label='Sell', zorder=5)
    plt.text(end_date, end_price, f' Sell\n{end_price}', color='red', fontsize=12, ha='left')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Trading Strategy Results')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
