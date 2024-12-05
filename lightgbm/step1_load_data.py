import numpy as np  # NumPyライブラリをインポート
from sklearn.datasets import (
    load_diabetes,
)  # 糖尿病データセットをロードする関数をインポート
import matplotlib.pyplot as plt  # グラフ描画ライブラリMatplotlibのPyplotモジュールをインポート
import seaborn as sns  # データ可視化ライブラリSeabornをインポート

# データセットの読み込み
diabetes = load_diabetes()  # 糖尿病データセットをロード
X = diabetes.data  # 特徴量データをXに格納
y = diabetes.target  # 目標変数データをyに格納

# 目標変数の分布を確認
sns.histplot(y, kde=True)  # 目標変数yのヒストグラムとカーネル密度推定を表示
plt.title("Distribution of Target Variable")  # グラフのタイトルを設定

plt.xlabel("Target Variable")  # X軸のラベルを設定
# 標変数は糖尿病データセットの場合、糖尿病の進行度合いを示す

plt.ylabel("Frequency")  # Y軸のラベルを設定
# 各バーの高さがその範囲内のデータポイントの数を示

plt.show()  # グラフを表示

# データの保存
np.save("X.npy", X)  # 特徴量データXを"X.npy"ファイルに保存
np.save("y.npy", y)  # 目標変数データyを"y.npy"ファイルに保存
