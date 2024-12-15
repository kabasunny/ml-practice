# model_training/train_model.py

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_model(features_df):
    # 説明変数と目的変数の分離
    X = features_df.drop('Label', axis=1)
    y = features_df['Label']

    # 学習データとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # データセットの作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # ハイパーパラメータの設定
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    # モデルの学習
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=100
    )

    # モデルの評価
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print('モデルの評価結果:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return gbm