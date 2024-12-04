import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# データの読み込み
X_selected = np.load("X_selected.npy")  # 選択された特徴量データを読み込む
y = np.load("y.npy")  # 目標変数データを読み込む

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)  # データを訓練セットとテストセットに分割

# 最終的なモデルの学習
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
model = lgb.train(
    {"objective": "regression", "metric": "rmse"},  # 基本的なハイパーパラメータ
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
)

# 特徴量重要度の取得
importance = model.feature_importance()
feature_names = model.feature_name()

# ターミナルに特徴量重要度を出力
for feature, imp in sorted(
    zip(feature_names, importance), key=lambda x: x[1], reverse=True
):
    print(f"Feature: {feature}, Importance: {imp}")

# 重要度が高い順にソートして上位10の特徴量を選択
top_10_features = sorted(
    zip(feature_names, importance), key=lambda x: x[1], reverse=True
)[:10]
top_10_feature_names = [feature for feature, _ in top_10_features]

print(f"Top 10 features: {top_10_feature_names}")

# 上位10の特徴量を基にしたデータセットの再構築
top_10_indices = [feature_names.index(name) for name in top_10_feature_names]
X_train_top_10 = X_train[:, top_10_indices]
X_test_top_10 = X_test[:, top_10_indices]

# データの保存
np.save("X_train_top_10.npy", X_train_top_10)  # 上位10の訓練データを保存
np.save("X_test_top_10.npy", X_test_top_10)  # 上位10のテストデータを保存

# 上位10の特徴量を使用して再度モデルを学習
train_data_top_10 = lgb.Dataset(X_train_top_10, label=y_train)
test_data_top_10 = lgb.Dataset(X_test_top_10, label=y_test, reference=train_data_top_10)
model_top_10 = lgb.train(
    {"objective": "regression", "metric": "rmse"},
    train_data_top_10,
    valid_sets=[test_data_top_10],
    num_boost_round=100,
)

# 最終的な予測と評価
y_pred_top_10 = model_top_10.predict(
    X_test_top_10, num_iteration=model_top_10.best_iteration
)
rmse_top_10 = root_mean_squared_error(y_test, y_pred_top_10)
print(f"Final RMSE with top 10 features: {rmse_top_10}")
