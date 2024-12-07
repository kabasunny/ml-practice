import yfinance as yf
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 株価データを取得
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    # print(data.head())
    '''
                  Open    High     Low   Close    Adj Close    Volume
    Date
    2023-12-01  2819.0  2842.0  2803.0  2833.0  2758.835693  26774000
    2023-12-04  2802.0  2802.5  2744.5  2767.5  2695.050293  30495700
    2023-12-05  2770.0  2784.5  2743.5  2753.5  2681.416748  24512600
    '''
    return data

# 日付情報を抽出し、週番号（Week）と年（Year）を追加
def preprocess_data(data): 
    data['Date'] = data.index
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Year'] = data['Date'].dt.isocalendar().year
    # print(data.head())
    '''
                  Open    High     Low   Close  ...    Volume       Date Week  Year
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ...  26774000 2023-12-01   48  2023     
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ...  30495700 2023-12-04   49  2023     
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ...  24512600 2023-12-05   49  2023     
    '''
    return data

# 各週ごとにランダムに選ばれた1日の始値を購入の基準とし、買いシグナルを生成
def random_daily_buy(data):
    # random.seed(42)  # 再現性のためにシードを設定
    data['Buy'] = False
    # print(data.head())
    '''
                  Open    High     Low   Close  ...       Date  Week  Year    Buy
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ... 2023-12-01    48  2023  False       
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ... 2023-12-04    49  2023  False       
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ... 2023-12-05    49  2023  False       
    2023-12-06  2770.5  2829.5  2758.0  2827.0  ... 2023-12-06    49  2023  False       
    2023-12-07  2800.0  2812.0  2776.0  2794.5  ... 2023-12-07    49  2023  False 
    '''

    grouped = data.groupby(['Year', 'Week'])
    for name, group in grouped:
        if not group.empty:
            random_day = group.sample(n=1).index
            data.loc[random_day, 'Buy'] = True
    # print(data.head())
    '''
                  Open    High     Low   Close  ...       Date  Week  Year    Buy       
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ... 2023-12-01    48  2023   True       
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ... 2023-12-04    49  2023  False       
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ... 2023-12-05    49  2023  False       
    2023-12-06  2770.5  2829.5  2758.0  2827.0  ... 2023-12-06    49  2023   True       
    2023-12-07  2800.0  2812.0  2776.0  2794.5  ... 2023-12-07    49  2023  False       
    
    '''

    buy_signals = data['Buy'].apply(lambda x: 'Buy' if x else 'Hold').tolist()
    # print(buy_signals)
    '''
    ['Buy', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold']     
    '''
    return buy_signals

# 売買の評価
def evaluate_trades(data, buy_signals, buy_threshold=0.1, sell_threshold=0.05):
    final_signals = []
    purchase_price = None
    trailing_stop_price = None
    profit_trades = 0
    trade_count = 0
    buy_indices = []
    sell_indices = []

    for i in range(len(buy_signals)):
        if buy_signals[i] == 'Buy' and purchase_price is None:
            purchase_price = data['Open'].iloc[i]  # 購入価格を設定
            trailing_stop_price = purchase_price * (1 - sell_threshold)
            trade_count += 1
            buy_indices.append(i)
            print(f"i:{i}, purchase_price:{purchase_price}, trailing_stop_price:{trailing_stop_price}, trade_count:{trade_count}")

        elif purchase_price is not None:
            current_price = data['Close'].iloc[i]
            
            # トレーリングストップの更新
            if current_price > purchase_price * (1 + buy_threshold):
                trailing_stop_price = current_price * (1 - sell_threshold)
                purchase_price = current_price
                print(f"i:{i}, updated purchase_price:{purchase_price}, trailing_stop_price:{trailing_stop_price}")
            
            # ロスカットまたはトレーリングストップで売却
            if current_price < trailing_stop_price:
                final_signals.append('Sell')
                if current_price > purchase_price:
                    profit_trades += 1
                sell_indices.append(i)
                print(f"i:{i}, Sell at current_price:{current_price}, profit_trades:{profit_trades}")
                purchase_price = None
                trailing_stop_price = None
            else:
                final_signals.append('Hold')
        else:
            final_signals.append('Hold')

    # 結果の評価
    success_ratio = profit_trades / trade_count if trade_count > 0 else 0
    print(f'Success Ratio: {success_ratio * 100:.2f}%')
    return final_signals, buy_indices, sell_indices

# 買いと売りのタイミングをチャートにプロット
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


# 使用例
symbol = "7203.T"  # トヨタ自動車のティッカーシンボル
start_date = "2023-10-01"
end_date = "2023-12-31"

data = fetch_stock_data(symbol, start_date, end_date)
data = preprocess_data(data)
buy_signals = random_daily_buy(data)

# 評価結果の取得
final_signals, buy_indices, sell_indices = evaluate_trades(data, buy_signals)
print(final_signals)

# 買いと売りのタイミングをチャートにプロット
plot_trades(data, buy_indices, sell_indices)