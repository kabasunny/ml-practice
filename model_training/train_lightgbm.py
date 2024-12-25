import lightgbm as lgb


def train_lightgbm(X_train, X_test, y_train, y_test):
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

    # LightGBMのデータセットを作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # モデルを訓練
    gbm = lgb.train(
        params,  # ハイパーパラメータ
        lgb_train,  # 訓練データ
        valid_sets=[lgb_eval],  # 検証データセット
        num_boost_round=1000,  # ブースティングの総ラウンド数
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100),
        ],  # アーリーストッピングとログ評価の設定
    )

    return gbm
