import pandas as pd
from model_training.plot_buy_chart import plot_results

def model_predict_and_plot(gbm, training_features_df, all_features_df, symbol_data_dict):
    # モデルの予測と結果の確認
    X_test = training_features_df.drop('Label', axis=1)
    y_test = training_features_df['Label']
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 結果をデータフレームにまとめ、シンボルカラムを追加して確認
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_binary,
        'Symbol': all_features_df['Symbol']
    }, index=training_features_df.index)

    print(results_df)

    # # 各シンボルごとのプロット
    # for symbol in symbol_data_dict.keys():
    #     daily_data = symbol_data_dict[symbol]
    #     features_df = all_features_df[all_features_df['Symbol'] == symbol]
    #     plot_results(daily_data, features_df, results_df, symbol)
