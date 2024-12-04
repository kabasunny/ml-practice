# diabetes_lightgbm.py
# 糖尿病データセットを使用して、特徴量を指定し糖尿病進行を予測

import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# データセットの読み込み
# load_diabetesを使用して糖尿病データセットを取得
diabetes = load_diabetes()
# 最初の2つの特徴量のみ使用
X = diabetes.data[:, :10]  # 特徴量の指定
# 1:年齢（Age）
# 2:性別（Sex）
# 3:BMI（Body Mass Index）
# 4:血圧（Average Blood Pressure）
# 5:血清1（S1）
# 6:血清2（S2）
# 7:血清3（S3）
# 8:血清4（S4）
# 9:血清5（S5）
# 10:血清6（S6）
y = diabetes.target

# 特徴量と目標値を表示
# print(f"X = diabetes.data[:, :2]:{X}")
# print(f"y = diabetes.target:{y}")

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
    "num_leaves": 60,  # 葉の数を設定
    "learning_rate": 0.1,  # 学習率を設定
    "feature_fraction": 0.8,  # 特徴量のサブサンプリング率を設定
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
