# california_housing_lightgbm_custom_features.py
# カリフォルニア住宅データセットでは、特定の特徴量を使用して住宅価格を予測

import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# データセットの読み込み
# fetch_california_housingを使用してカリフォルニア住宅データセットを取得
housing = fetch_california_housing()
# 特定の特徴量（1列目、3列目、5列目）と目標値（target）を取得
X = housing.data[:, [0, 2, 4]]
# X = housing.data 全データ取得
# 1:MedInc（Median Income）: ブロックグループ内の世帯収入の中央値（単位：10,000ドル）
# 2:HouseAge（House Age）: ブロックグループ内の住宅の築年数の中央値
# 3:AveRooms（Average Rooms）: 各住宅における部屋の平均数
# 4:AveBedrms（Average Bedrooms）: 各住宅におけるベッドルームの平均数
# 5:Population（Population）: ブロックグループ内の人口
# 6:AveOccup（Average Occupancy）: 各住宅における住人の平均数
# 7:Latitude（Latitude）: ブロックグループの緯度
# 8:Longitude（Longitude）: ブロックグループの経度
y = housing.target

# 特徴量と目標値を表示
# print(f"X = housing.data[:, [0, 2, 4]]:{X}")
# print(f"y = housing.target:{y}")

# 訓練データとテストデータに分割
# train_test_splitを使用してデータを訓練データ（80%）とテストデータ（20%）に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM用データセットの作成
# lgb.Datasetを使用して訓練データとテストデータをLightGBM用に変換
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# モデルのパラメータ設定
# LightGBMモデルのパラメータを設定
params = {
    "objective": "regression",  # 回帰問題として設定
    "metric": "rmse",  # 評価指標をRMSEに設定
    "boosting_type": "gbdt",  # 勾配ブースティングを使用
    "num_leaves": 31,  # 葉の数を設定
    "learning_rate": 0.05,  # 学習率を設定
    "feature_fraction": 0.9,  # 特徴量のサブサンプリング率を設定
}

# Early stoppingの設定
# 早期終了のためのコールバックを設定
callbacks = [lgb.early_stopping(stopping_rounds=10)]

# モデルの学習
print("Training model...")
# lgb.trainを使用してモデルを学習
model = lgb.train(
    params, train_data, valid_sets=[test_data], num_boost_round=100, callbacks=callbacks
)

# 予測
# 学習したモデルを使用してテストデータに対する予測を行う
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 評価
# 平均二乗平方根誤差（RMSE）を計算してモデルの性能を評価
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")
