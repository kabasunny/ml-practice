import matplotlib.pyplot as plt

# グラフ描画関数
def plot_cycles(data, peaks, troughs, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label="Close Price", color="blue")
    plt.scatter(data.index[peaks], data.iloc[peaks], color="red", label="Peaks")
    plt.scatter(data.index[troughs], data.iloc[troughs], color="green", label="Troughs")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
