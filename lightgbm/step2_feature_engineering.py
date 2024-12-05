import numpy as np  # NumPyライブラリをインポート
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    PolynomialFeatures,
)  # データ変換と前処理のための関数をインポート

# データの読み込み
X = np.load("X.npy")  # 保存された特徴量データを"X.npy"ファイルから読み込む


def engineer_features(X):
    X_engineered = np.copy(X)  # 特徴量データのコピーを作成

    # 多項式特徴量と交互作用特徴量を追加（degree=2に制限）
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False
    )  # 2次の多項式特徴量と交互作用項を生成
    X_poly = poly.fit_transform(
        X_engineered[:, :10]
    )  # 元の特徴量から多項式特徴量を生成
    X_engineered = np.concatenate(
        [X_engineered, X_poly], axis=1
    )  # 元の特徴量と生成した多項式特徴量を結合
    # 元の特徴量 10個 + 新たに生成される交互作用項 45個 = 55個（新たに生成される特徴量の合計）

    # その他の特徴量
    X_engineered = np.column_stack(
        (X_engineered, X[:, 2] * X[:, 3])
    )  # BMIと血圧の積を新たな特徴量として追加
    X_engineered = np.column_stack(
        (X_engineered, np.mean(X[:, 4:10], axis=1))
    )  # 血清値の平均を新たな特徴量として追加
    X_engineered = np.column_stack(
        (X_engineered, X[:, 0] ** 2, X[:, 2] ** 2)
    )  # 年齢とBMIの二乗項を新たな特徴量として追加

    # 特徴量の変換
    scaler = StandardScaler()  # 標準化のためのスケーラーを作成
    X_engineered = scaler.fit_transform(X_engineered)  # 特徴量を標準化
    pt = (
        PowerTransformer()
    )  # データの正規性を向上させるためのパワートランスフォーマーを作成
    X_engineered = pt.fit_transform(X_engineered)  # 特徴量を変換

    return X_engineered  # 変換された特徴量データを返す


X_engineered = engineer_features(
    X
)  # 特徴量エンジニアリング関数を呼び出し、変換された特徴量データを取得

# 入力データの形状を確認
print(f"Original shape: {X.shape}")  # 元の特徴量データの形状を表示
print(f"Engineered shape: {X_engineered.shape}")  # 変換された特徴量データの形状を表示

# データの保存
np.save(
    "X_engineered.npy", X_engineered
)  # 変換された特徴量データを"X_engineered.npy"ファイルに保存
