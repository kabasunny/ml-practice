# titanic_lightgbm.py
# タイタニックの生存予測モデルをLightGBMを使用して構築

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの読み込み
csv_file = "titanic.csv"

if not os.path.exists(csv_file):
    # CSVファイルが存在しない場合、インターネットからデータを取得
    print(f"{csv_file} が見つかりません。データをインターネットから取得します。")
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    data = pd.read_csv(url)
    data.to_csv(csv_file, index=False)
else:
    # CSVファイルが存在する場合、そのファイルを読み込む
    print(f"{csv_file} を読み込みます。")
    data = pd.read_csv(csv_file)

# データの前処理
# 性別を数値化
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
# 乗船港を数値化
data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# 数値列の欠損値を平均値で補完
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# テキスト列の欠損値をモード（最頻値）で補完
categorical_columns = data.select_dtypes(include=["object"]).columns
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# 特徴量と目標変数を設定
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data[features]
y = data["Survived"]

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM用データセットの作成
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# モデルのパラメータ設定
params = {
    "objective": "binary",  # 二値分類
    "metric": "binary_error",  # 評価指標をエラー率に設定
    "boosting_type": "gbdt",  # 勾配ブースティングを使用
    "num_leaves": 31,  # 葉の数を設定
    "learning_rate": 0.05,  # 学習率を設定
    "feature_fraction": 0.9,  # 特徴量のサブサンプリング率を設定
}

# Early stoppingの設定
callbacks = [lgb.early_stopping(stopping_rounds=10)]

# モデルの学習
print("Training model...")
model = lgb.train(
    params, train_data, valid_sets=[test_data], num_boost_round=100, callbacks=callbacks
)

# 予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# 0.5を閾値として二値化
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

# 評価
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")
