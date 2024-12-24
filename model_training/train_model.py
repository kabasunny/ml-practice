# model_training\train_model.py
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import early_stopping, log_evaluation
import pandas as pd
from model_training.plot_buy_chart import (
    plot_feature_importance,
)  # 新しいプロット関数をインポート


def train_and_evaluate_model(features_df):

    # 不正解ラベルのデータ数と正解ラベルのデータ数を表示
    label_counts = features_df["Label"].value_counts()
    print(f"不正解ラベル数: {label_counts[0]}")
    print(f"正解ラベル数  : {label_counts[1]}")

    # 説明変数（特徴量）と目的変数（ラベル）の分離
    X = features_df.drop("Label", axis=1)  # 説明変数
    y = features_df["Label"]  # 目的変数

    # SMOTEによるオーバーサンプリング
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # SMOTE後のラベル数を表示
    label_counts_resampled = pd.Series(y_res).value_counts()
    print(f"SMOTE後の不正解ラベル数: {label_counts_resampled[0]}")
    print(f"SMOTE後の正解ラベル数  : {label_counts_resampled[1]}")

    # データセットを訓練データとテストデータに分割（80%を訓練データ、20%をテストデータ）
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, shuffle=True, random_state=42
    )

    # 学習データ数とテストデータ数を表示
    print(f"X_train: 訓練データの特徴量（説明変数）  : {len(X_train)}")
    print(
        f"y_train: 訓練データのラベル（目的変数）  : {len(y_train)}"
    )  # このラベルを用いてモデルを訓練
    print(f"X_test : テストデータの特徴量（説明変数）: {len(X_test)}")
    print(
        f"y_test : テストデータのラベル（目的変数）: {len(y_test)}"
    )  # このラベルを用いてモデルの予測性能を評価

    # LightGBMのデータセットを作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
    "objective": "binary",  # バイナリ分類問題を指定
    "metric": "binary_logloss",  # 損失関数としてバイナリログ損失を使用
    "boosting_type": "gbdt",  # 勾配ブースティング決定木を使用
    "learning_rate": 0.01,  # 学習率を設定（小さい値にすることでモデルの安定性を向上）
    "num_leaves": 51,  # リーフノードの数を設定（大きい値にすることでモデルの複雑さを増加）
    "max_depth": -1,  # 決定木の最大深さを設定（-1は無制限を意味する）
    "min_data_in_leaf": 10,  # 各リーフノードに含まれる最小データ数を設定（過学習を防ぐために使用）
    "feature_fraction": 0.8,  # 各ブーストラウンドで使用する特徴量の割合を設定（過学習を防ぐために使用）
    "bagging_fraction": 0.8,  # 各ブーストラウンドで使用するデータの割合を設定（過学習を防ぐために使用）
    "bagging_freq": 10,  # バギングを実行する頻度を設定（bagging_fractionと組み合わせて使用）
    "verbose": -1,  # 詳細出力を抑制（0や1に設定すると詳細出力が有効になる）
    }

    # モデルを訓練
    gbm = lgb.train(
        params,  # ハイパーパラメータ
        lgb_train,  # 訓練データ
        valid_sets=[lgb_eval],  # 検証データセット
        num_boost_round=1000,  # ブースティングの総ラウンド数
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(100),
        ],  # アーリーストッピングとログ評価の設定
    )

    # テストデータに対する予測を行い、0と1に変換
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # モデルの評価指標を計算
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    # Recallのデバック
    print(f"train_model.py [len(y_test):{len(y_test)}]")
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    # 評価結果を表示
    print("モデルの評価結果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 特徴量重要度の取得と表示
    feature_importance = gbm.feature_importance()
    feature_names = X_train.columns
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # 特徴量重要度をターミナルに表示
    print("特徴量の重要度:")
    print(importance_df)

    # # 特徴量重要度のプロット
    # plot_feature_importance(importance_df)

    return gbm
