import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data['Date'] = data.index
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    
    # ターゲットは次の日の終値が上昇するかどうか（1: 上昇, 0: 下降）
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # 欠損値を削除
    data = data.dropna()
    
    # 特徴量とターゲットを分割
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day', 'Weekday', 'Month', 'Year']
    X = data[features]
    y = data['Target']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # コールバック関数で early stopping を設定
    callbacks = [lgb.early_stopping(stopping_rounds=10)]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=1000,
        callbacks=callbacks
    )
    
    return model, X_test, y_test

def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)  # 予測を0または1に変換
    
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'Accuracy: {accuracy}')
    
    return y_pred_binary

def generate_trading_signals(predictions, X_test, data):
    signals = []
    for i, pred in enumerate(predictions):
        if pred == 1 and X_test.iloc[i]['Weekday'] == 0:  # 週初めの始値で買判断
            signals.append('Buy')
        elif pred == 0:
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals

def risk_management(signals, close_prices, buy_threshold=0.1, sell_threshold=0.05):
    final_signals = []
    trailing_stop_price = None
    for i in range(len(signals)):
        if signals[i] == 'Buy':
            if trailing_stop_price is None:
                trailing_stop_price = close_prices[i] * (1 - sell_threshold)
            elif close_prices[i] > trailing_stop_price * (1 + buy_threshold):
                trailing_stop_price = close_prices[i] * (1 - sell_threshold)
            final_signals.append('Buy')
        elif signals[i] == 'Sell':
            if trailing_stop_price is not None and close_prices[i] < trailing_stop_price:
                final_signals.append('Sell')
                trailing_stop_price = None  # リセット
            else:
                final_signals.append('Hold')
        else:
            final_signals.append('Hold')
    return final_signals

# 使用例
symbol = "^TOPX"  # TOPIXのティッカーシンボル
start_date = "2020-01-01"
end_date = "2023-01-01"

data = fetch_stock_data(symbol, start_date, end_date)
X, y = preprocess_data(data)
model, X_test, y_test = train_model(X, y)
predictions = predict_and_evaluate(model, X_test, y_test)

# 売買判断の生成
signals = generate_trading_signals(predictions, X_test, data)
close_prices = X_test['Close'].values

# リスク管理
final_signals = risk_management(signals, close_prices)
print(final_signals)