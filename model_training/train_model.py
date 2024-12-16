from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import early_stopping, log_evaluation

def train_and_evaluate_model(features_df):
    # 説明変数（特徴量）と目的変数（ラベル）の分離
    X = features_df.drop('Label', axis=1)  # 説明変数
    y = features_df['Label']  # 目的変数

    # SMOTEによるオーバーサンプリング
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # データセットを訓練データとテストデータに分割（80%を訓練データ、20%をテストデータ）
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=True, random_state=42)

    # LightGBMのデータセットを作成
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBMのハイパーパラメータを設定
    params = {
        'objective': 'binary',  # バイナリ分類問題を指定
        'metric': 'binary_logloss',  # 損失関数としてバイナリログ損失を使用
        'boosting_type': 'gbdt',  # 勾配ブースティング決定木を使用
        'learning_rate': 0.05,  # 学習率を設定
        'num_leaves': 31,  # リーフノードの数を設定
        'verbose': -1  # 詳細出力を抑制
    }

    # モデルを訓練
    gbm = lgb.train(
        params,  # ハイパーパラメータ
        lgb_train,  # 訓練データ
        valid_sets=[lgb_eval],  # 検証データセット
        num_boost_round=1000,  # ブースティングの総ラウンド数
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]  # アーリーストッピングとログ評価の設定
    )

    # テストデータに対する予測を行い、0と1に変換
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # モデルの評価指標を計算
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    # 評価結果を表示
    print('モデルの評価結果:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return gbm
