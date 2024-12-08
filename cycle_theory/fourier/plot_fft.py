import matplotlib.pyplot as plt

# グラフ描画関数
def plot_fft(fft_period, fft_amplitude, title):
    plt.figure(figsize=(12, 6))
    plt.plot(fft_period, fft_amplitude, label=title)
    plt.xscale("log")
    plt.xlabel("Period (Days)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
