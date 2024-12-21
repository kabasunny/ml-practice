import matplotlib.pyplot as plt

def plot_cycles(data, peaks, troughs, title):
    plt.figure(figsize=(14, 7))
    # 'Low'列の代わりに実際の列名を使用
    plt.plot(data.index, data, label='Detrended Prices')
    
    # PeaksとTroughsをプロットする前にサイズをチェック
    if len(peaks) > 0:
        plt.scatter(data.index[peaks], data.iloc[peaks], color='red', label='Peaks')
    if len(troughs) > 0:
        plt.scatter(data.index[troughs], data.iloc[troughs], color='blue', label='Troughs')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Detrended Prices')
    plt.legend()
    plt.show()