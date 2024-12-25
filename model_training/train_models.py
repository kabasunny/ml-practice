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
from model_training.evaluate_model import evaluate_model


# LightGBMのトレーニング
def train_lightgbm(X_train, X_test, y_train, y_test):
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
    # モデルの評価
    evaluate_model(gbm, X_test, y_test)
    return gbm


# ランダムフォレストのトレーニング
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# XGBoostのトレーニング
def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# CatBoostのトレーニング
def train_catboost(X_train, X_test, y_train, y_test):
    model = CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, verbose=0)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# AdaBoostのトレーニング
def train_adaboost(X_train, X_test, y_train, y_test):
    model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# Gradient Boosting Machines (GBM)のトレーニング
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# サポートベクターマシンのトレーニング
def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# k近傍法のトレーニング
def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# ロジスティック回帰のトレーニング
def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    return model


# モデルタイプに応じてトレーニング関数を選択
def train_model(model_type, X_train, X_test, y_train, y_test):
    if model_type == "lightgbm":
        return train_lightgbm(X_train, X_test, y_train, y_test)
    elif model_type == "rand_frst":
        return train_random_forest(X_train, X_test, y_train, y_test)
    elif model_type == "xgboost":
        return train_xgboost(X_train, X_test, y_train, y_test)
    elif model_type == "catboost":
        return train_catboost(X_train, X_test, y_train, y_test)
    elif model_type == "adaboost":
        return train_adaboost(X_train, X_test, y_train, y_test)
    elif model_type == "grdt_bstg":
        return train_gradient_boosting(X_train, X_test, y_train, y_test)
    elif model_type == "svm":
        return train_svm(X_train, X_test, y_train, y_test)
    elif model_type == "knn":
        return train_knn(X_train, X_test, y_train, y_test)
    elif model_type == "logc_regr":
        return train_logistic_regression(X_train, X_test, y_train, y_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
