from data_processing.fetch_stock_data import fetch_stock_data
from data_processing.detrend_prices import detrend_prices
from cycle_theory.peak_trough.detect_cycles import detect_cycles
from cycle_theory.fourier.fft_analysis import fft_analysis
from features.feature_engineering import create_features
from labels.create_labels import create_labels
from model_training.train_model import train_and_evaluate_model
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tkinter')


def main():
    symbol = "7203.T"
    trade_start_date = pd.Timestamp("2014-08-01")
    period_days = 365 * 5

    start_date = trade_start_date - pd.Timedelta(days=period_days)
    end_date = trade_start_date + pd.Timedelta(days=period_days)

    # データの取得
    daily_data = fetch_stock_data(symbol, start_date, end_date)
    # print(f'daily_data : {daily_data}')
    # print(f'weekly_data : {weekly_data}')
    # print(f'monthly_data : {monthly_data}')
    
    # ラベルの作成
    labeled_data = create_labels(daily_data)
    # print(f'labels : {labels}')


    
    # 以下はコメントアウトされたコードです
    # # トレンド除去
    # remove_trend = True
    # detrended_prices = detrend_prices(daily_data, remove_trend)
    # detrended_weekly_prices = detrend_prices(weekly_data, remove_trend)
    # detrended_monthly_prices = detrend_prices(monthly_data, remove_trend)

    # # 特徴量の作成
    # features = []
    # for date in weekly_data.index:
    #     feature = create_features(date, detrended_prices, detrended_weekly_prices, detrended_monthly_prices)
    #     features.append(feature)

    # # 特徴量のデータフレーム化
    # features_df = pd.DataFrame(features, index=weekly_data.index)

    # # ラベルを結合
    # features_df['Label'] = labels

    # # 欠損値の削除
    # features_df.dropna(inplace=True)

    # # モデルの学習と評価
    # gbm = train_and_evaluate_model(features_df)

if __name__ == "__main__":
    main()
