import numpy as np  # NumPyライブラリをインポート
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
)  # 特徴量選択のための関数をインポート
from sklearn.model_selection import cross_val_score  # 交差検証のための関数をインポート
import lightgbm as lgb  # LightGBMライブラリをインポート
from sklearn.datasets import load_diabetes  # データセットの読み込み

# データの読み込み
X_engineered = np.load(
    "X_engineered.npy"
)  # 変換された特徴量データを"X_engineered.npy"ファイルから読み込む
y = np.load("y.npy")  # 目標変数データを"y.npy"ファイルから読み込む

best_k = 30  # 初期値としてのk（選択する特徴量の数）
best_rmse = float("inf")  # 初期値としての最良RMSE（無限大）

# 最適なkを見つけるためのループ
for k in range(10, 40, 5):  # kの値を10から5ずつ増加させて40未満まで試行
    selector = SelectKBest(
        score_func=f_regression, k=k
    )  # f_regressionスコア関数を使用してk個の特徴量を選択
    X_selected = selector.fit_transform(
        X_engineered, y
    )  # 特徴量選択を実行し、選択された特徴量データを取得
    scores = cross_val_score(
        lgb.LGBMRegressor(), X_selected, y, cv=5, scoring="neg_root_mean_squared_error"
    )  # 交差検証を実行してRMSEを計算
    rmse = -scores.mean()  # RMSEの平均を取得
    print(f"k={k}, RMSE={rmse}")  # kと対応するRMSEを出力
    if rmse < best_rmse:  # 新しいRMSEが現在の最良RMSEよりも小さい場合
        best_rmse = rmse  # 最良RMSEを更新
        best_k = k  # 最適なkを更新

selector = SelectKBest(score_func=f_regression, k=best_k)  # 最良のkで特徴量選択を再実行
X_selected = selector.fit_transform(
    X_engineered, y
)  # 特徴量選択を実行し、最終的な選択された特徴量データを取得

# 入力データの形状を確認
print(
    f"Selected features shape: {X_selected.shape}"
)  # 選択された特徴量データの形状を表示

# 選択された特徴量のインデックスを取得
selected_indices = selector.get_support(indices=True)

# 元の特徴量名を取得（糖尿病データセットの場合）
diabetes = load_diabetes()
feature_names = diabetes.feature_names + [
    f"poly_feature_{i}"
    for i in range(X_engineered.shape[1] - len(diabetes.feature_names))
]

# 選択された特徴量の名前を表示
selected_feature_names = [feature_names[i] for i in selected_indices]
print(f"Selected features: {selected_feature_names}")

# データの保存
np.save(
    "X_selected.npy", X_selected
)  # 最終的に選択された特徴量データを"X_selected.npy"ファイルに保存
