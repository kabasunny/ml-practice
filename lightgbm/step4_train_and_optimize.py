import numpy as np  # NumPyライブラリをインポート
from sklearn.model_selection import (
    train_test_split,
)  # データ分割のための関数をインポート
import lightgbm as lgb  # LightGBMライブラリをインポート
from sklearn.metrics import root_mean_squared_error  # 評価指標のための関数をインポート
import optuna  # ハイパーパラメータ最適化ライブラリをインポート

# データの読み込み
X_selected = np.load(
    "X_selected.npy"
)  # 選択された特徴量データを"X_selected.npy"ファイルから読み込む
y = np.load("y.npy")  # 目標変数データを"y.npy"ファイルから読み込む

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)  # データを訓練セットとテストセットに分割（20%をテストデータ）


# LightGBMの目的関数
def objective(trial):
    # ハイパーパラメータの設定
    params = {
        "objective": "regression",  # 回帰タスクの目的関数
        "metric": "rmse",  # 評価指標としてRMSEを使用
        "boosting_type": "gbdt",  # 勾配ブースティング決定木を使用
        "num_leaves": trial.suggest_int(
            "num_leaves", 10, 100
        ),  # num_leavesを10から100の範囲で提案
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),  # learning_rateを対数スケールで提案
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.4, 1.0
        ),  # feature_fractionを0.4から1.0の範囲で提案
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 2, 20
        ),  # min_data_in_leafを2から20の範囲で提案
        "max_depth": trial.suggest_int(
            "max_depth", 3, 15
        ),  # max_depthを3から15の範囲で提案
        "lambda_l1": trial.suggest_float(
            "lambda_l1", 1e-8, 10.0, log=True
        ),  # L1正則化の強度を対数スケールで提案
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 1e-8, 10.0, log=True
        ),  # L2正則化の強度を対数スケールで提案
    }

    # LightGBM用データセットの作成
    train_data = lgb.Dataset(X_train, label=y_train)  # 訓練データセットの作成
    test_data = lgb.Dataset(
        X_test, label=y_test, reference=train_data
    )  # テストデータセットの作成

    # モデルの学習
    model = lgb.train(
        params,  # ハイパーパラメータ
        train_data,  # 訓練データセット
        valid_sets=[test_data],  # 検証用データセット
        num_boost_round=500,  # 最大500回のブースティングラウンド
        callbacks=[
            lgb.early_stopping(stopping_rounds=10)
        ],  # 早期終了の設定（改善が見られない場合10回で停止）
    )

    # 予測
    y_pred = model.predict(
        X_test, num_iteration=model.best_iteration
    )  # テストデータに対する予測

    # 評価指標の計算
    rmse = root_mean_squared_error(y_test, y_pred)  # RMSEの計算
    return rmse  # RMSEを返す


# Optunaによるハイパーパラメータ最適化
sampler = optuna.samplers.TPESampler(multivariate=True)  # TPEサンプラーを使用
study = optuna.create_study(
    direction="minimize", sampler=sampler
)  # 最小化を目指すスタディの作成
study.optimize(objective, n_trials=100)  # 100回の試行で最適化を実行

# ベストパラメータの表示
print(f"Best hyperparameters: {study.best_params}")  # 最適なハイパーパラメータを表示

# 最終的なモデルの学習
best_params = study.best_params  # 最適なパラメータを取得
train_data = lgb.Dataset(X_train, label=y_train)  # 訓練データセットの作成
test_data = lgb.Dataset(
    X_test, label=y_test, reference=train_data
)  # テストデータセットの作成
model = lgb.train(
    best_params,  # 最適なパラメータ
    train_data,  # 訓練データセット
    valid_sets=[test_data],  # 検証用データセット
    num_boost_round=500,  # 最大500回のブースティングラウンド
    callbacks=[lgb.early_stopping(stopping_rounds=10)],  # 早期終了の設定
)

# 最終的な予測と評価
y_pred = model.predict(
    X_test, num_iteration=model.best_iteration
)  # テストデータに対する予測
rmse = root_mean_squared_error(y_test, y_pred)  # RMSEの計算
print(f"Final RMSE: {rmse}")  # 最終的なRMSEを表示

# 特徴量重要度の取得と表示
import matplotlib.pyplot as plt  # プロットライブラリをインポート

# 特徴量重要度の取得
importance = model.feature_importance()
feature_names = model.feature_name()

# ターミナルに特徴量重要度を出力
for feature, imp in sorted(
    zip(feature_names, importance), key=lambda x: x[1], reverse=True
):
    print(f"Feature: {feature}, Importance: {imp}")

# 特徴量重要度のプロット
lgb.plot_importance(model, max_num_features=20)  # 重要度の高い上位20の特徴量をプロット
plt.show()  # プロットを表示
