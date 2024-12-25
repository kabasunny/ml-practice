from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# LightGBMのトレーニング
def train_lightgbm(X_train, X_test, y_train, y_test):
    """
    LightGBM: 勾配ブースティング決定木を使用してバイナリ分類を行うモデル。
    高速な学習と高い予測精度を特徴とする。
    """
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 10,
        "verbose": -1,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )
    return gbm


# ランダムフォレストのトレーニング
def train_random_forest(X_train, y_train):
    """
    Random Forest: 多数の決定木を使ったアンサンブル学習モデル。
    過学習を抑制し、変数の重要度を評価できる。
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# XGBoostのトレーニング
def train_xgboost(X_train, y_train):
    """
    XGBoost: 勾配ブースティングフレームワークを使用した高性能な学習アルゴリズム。
    高い予測精度と高速な学習を特徴とする。
    """
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model


# CatBoostのトレーニング
def train_catboost(X_train, y_train):
    """
    CatBoost: カテゴリデータの処理に優れた勾配ブースティングモデル。
    高精度で過学習を抑制する。
    """
    model = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, verbose=0)
    model.fit(X_train, y_train)
    return model


# AdaBoostのトレーニング
def train_adaboost(X_train, y_train):
    """
    AdaBoost: 弱学習器（通常は決定木）を使用したアンサンブル学習モデル。
    ノイズに対して頑健で高精度。
    """
    model = AdaBoostClassifier(
        n_estimators=100, algorithm="SAMME", random_state=42
    )  # SAMME.R は確率を使う デフォルトのSAMME.Rアルゴリズムが将来的に削除される予定であり、明示的にSAMMEアルゴリズム
    model.fit(X_train, y_train)
    return model


# Gradient Boosting Machines (GBM)のトレーニング
def train_gradient_boosting(X_train, y_train):
    """
    Gradient Boosting Machines (GBM): 勾配ブースティングを使用したモデル。
    高精度と柔軟な設定を特徴とする。
    """
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# サポートベクターマシンのトレーニング
def train_svm(X_train, y_train):
    """
    Support Vector Machine (SVM): 高次元空間でデータを分離する最適なハイパープレーンを見つけるモデル。
    効率的に分類問題を解くことができる。
    """
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


# k近傍法のトレーニング
def train_knn(X_train, y_train):
    """
    K-Nearest Neighbors (KNN): データポイントの多数決によって分類を行うモデル。
    直感的でシンプルなアルゴリズム。
    """
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


# ロジスティック回帰のトレーニング
def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression: ロジスティック関数を用いた線形モデル。
    確率を出力し、2値分類に適している。
    """
    model = LogisticRegression(
        random_state=42, max_iter=1000
    )  # イテレーション数を増やした デフォルトは100
    model.fit(X_train, y_train)
    return model


# モデルタイプに応じてトレーニング関数を選択
def train_model(model_type, X_train, X_test, y_train, y_test):
    if model_type == "lightgbm":
        return train_lightgbm(X_train, X_test, y_train, y_test)
    elif model_type == "random_forest":
        return train_random_forest(X_train, y_train)
    elif model_type == "xgboost":
        return train_xgboost(X_train, y_train)
    elif model_type == "catboost":
        return train_catboost(X_train, y_train)
    elif model_type == "adaboost":
        return train_adaboost(X_train, y_train)
    elif model_type == "gradient_boosting":
        return train_gradient_boosting(X_train, y_train)
    elif model_type == "svm":
        return train_svm(X_train, y_train)
    elif model_type == "knn":
        return train_knn(X_train, y_train)
    elif model_type == "logistic_regression":
        return train_logistic_regression(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
