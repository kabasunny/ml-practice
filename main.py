from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from features.create_features import create_features
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
import pandas as pd
import matplotlib.pyplot as plt
from plot_buy_signals import plot_buy_signals  # 新しいファイルからインポート

def main():
    symbol = "7203.T"
    trade_start_date = pd.Timestamp("2014-08-01")
    period_days = 365 * 2

    start_date = trade_start_date - pd.Timedelta(days=period_days)
    end_date = trade_start_date + pd.Timedelta(days=period_days)

    # データの取得
    daily_data = fetch_stock_data(symbol, start_date, end_date)
    weekly_data = daily_data.resample("W").ffill()
    monthly_data = daily_data.resample("ME").ffill()

    # ラベルの作成
    labeled_data = create_labels(daily_data)

    # トレンド除去
    remove_trend = True
    detrended_prices = detrend_prices(daily_data, remove_trend)
    detrended_weekly_prices = detrend_prices(weekly_data, remove_trend)
    detrended_monthly_prices = detrend_prices(monthly_data, remove_trend)

    # 特徴量の作成
    features = []
    for date in daily_data.index:
        feature = create_features(date, detrended_prices, detrended_weekly_prices, detrended_monthly_prices)
        features.append(feature)

    # 特徴量のデータフレーム化
    features_df = pd.DataFrame(features, index=daily_data.index)

    # ラベルを結合
    features_df['Label'] = labeled_data['Label']
    
    # 欠損値の削除
    features_df.dropna(inplace=True)

    # ラベルのユニークな値とそのカウントを確認
    print(features_df['Label'].value_counts())

    # チャートを表示、正解ラベルにマーク
    plot_buy_signals(daily_data, features_df, symbol)

    # モデルの学習と評価
    # gbm = train_and_evaluate_model(features_df)

if __name__ == "__main__":
    main()