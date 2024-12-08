from fetch_stock_data import fetch_stock_data
from detrend_prices import detrend_prices
from detect_cycles import detect_cycles
from plot_cycles import plot_cycles

# メイン処理
def main():
    symbol = "7203.T"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # データの取得（日足）
    data = fetch_stock_data(symbol, start_date, end_date)  # pandas を必要とする

    # 前処理（線形トレンド除去）
    detrended_prices = detrend_prices(data)

    # サイクルの検出（日足）
    peaks, troughs, avg_peak_cycle, avg_trough_cycle = detect_cycles(detrended_prices)
    print(f"平均ピークサイクル: {avg_peak_cycle} 日")
    print(f"平均谷サイクル: {avg_trough_cycle} 日")

    # チャートの描画（日足）
    plot_cycles(detrended_prices, peaks, troughs, f"Daily Cycle Analysis for {symbol}")  # pandas を必要とする

    # データのリサンプリング関数（週足）
    weekly_data = detrended_prices.resample("W").ffill()  # 週足にリサンプリングし、前週の値を埋める（pandas を必要とする）

    # サイクルの検出（週足）
    peaks, troughs, avg_peak_cycle, avg_trough_cycle = detect_cycles(weekly_data)
    print(f"平均ピークサイクル: {avg_peak_cycle} 週")
    print(f"平均谷サイクル: {avg_trough_cycle} 週")

    # チャートの描画（週足）
    plot_cycles(weekly_data, peaks, troughs, f"Weekly Cycle Analysis for {symbol}")  # pandas を必要とする

    # データのリサンプリング関数（月足）
    monthly_data = detrended_prices.resample("ME").ffill()  # 月足にリサンプリングし、前月の値を埋める（pandas を必要とする）

    # サイクルの検出（月足）
    peaks, troughs, avg_peak_cycle, avg_trough_cycle = detect_cycles(monthly_data)
    print(f"平均ピークサイクル: {avg_peak_cycle} ヶ月")
    print(f"平均谷サイクル: {avg_trough_cycle} ヶ月")

    # チャートの描画（月足）
    plot_cycles(monthly_data, peaks, troughs, f"Monthly Cycle Analysis for {symbol}")  # pandas を必要とする

if __name__ == "__main__":
    main()
