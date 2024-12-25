from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd


def prepare_data(features_df):
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

    return X_train, X_test, y_train, y_test
