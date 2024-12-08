from fetch_stock_data import fetch_stock_data
from detrend_prices import detrend_prices
from fft_analysis import fft_analysis
from plot_fft import plot_fft

# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # データの取得
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas を必要とする

    # 前処理（線形トレンド除去）
    detrended_prices = detrend_prices(data)

    # FFTによるサイクル解析
    fft_period, fft_amplitude, dominant_periods = fft_analysis(detrended_prices.values)
    print("上位5つの周期:", dominant_periods)

    # FFTスペクトルのプロット
    plot_fft(fft_period, fft_amplitude, "FFT Analysis (Log Scale)")

if __name__ == "__main__":
    main()
