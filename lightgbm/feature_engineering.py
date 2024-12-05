import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, make_scorer
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

# データセットの読み込み
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 目標変数の分布を確認
sns.histplot(y, kde=True)
plt.show()


# 特徴量エンジニアリング
def engineer_features(X):
    X_engineered = np.copy(X)

    # 多項式特徴量と交互作用特徴量を追加（degree=2に制限）
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_engineered[:, :10])
    X_engineered = np.concatenate([X_engineered, X_poly], axis=1)

    # その他の特徴量
    X_engineered = np.column_stack((X_engineered, X[:, 2] * X[:, 3]))  # BMIと血圧の積
    X_engineered = np.column_stack(
        (X_engineered, np.mean(X[:, 4:10], axis=1))
    )  # 血清値の平均
    X_engineered = np.column_stack(
        (X_engineered, X[:, 0] ** 2, X[:, 2] ** 2)
    )  # 年齢とBMIの二乗項

    # 特徴量の変換
    scaler = StandardScaler()
    X_engineered = scaler.fit_transform(X_engineered)
    pt = PowerTransformer()
    X_engineered = pt.fit_transform(X_engineered)

    return X_engineered


X_engineered = engineer_features(X)

# 特徴量選択
best_k = 30  # 初期値
best_rmse = float("inf")

# 最適なkを見つけるためのループ
for k in range(10, 40, 5):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_engineered, y)
    scores = cross_val_score(
        lgb.LGBMRegressor(), X_selected, y, cv=5, scoring="neg_root_mean_squared_error"
    )
    rmse = -scores.mean()
    print(f"k={k}, RMSE={rmse}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_k = k

selector = SelectKBest(score_func=f_regression, k=best_k)
X_engineered = selector.fit_transform(X_engineered, y)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)


# LightGBMの目的関数
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
        "max_depth": trial.suggest_int("max_depth", 3, 15),  # max_depthを15に拡大
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    # LightGBM用データセットの作成
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # モデルの学習
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    # 予測
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 評価指標の計算
    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse


# Optunaによるハイパーパラメータ最適化
sampler = optuna.samplers.TPESampler(multivariate=True)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100)

# ベストパラメータの表示
print(f"Best hyperparameters: {study.best_params}")

# 最終的なモデルの学習
best_params = study.best_params
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
model = lgb.train(
    best_params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=10)],
)

# 最終的な予測と評価
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Final RMSE: {rmse}")

# 特徴量重要度のプロット
lgb.plot_importance(model, max_num_features=20)
plt.show()
