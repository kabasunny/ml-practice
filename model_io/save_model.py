import joblib
import os


# モデルの保存
def save_model(model, model_type, directory="trained_models"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"trained_model_{model_type}.pkl"
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    # print(f"モデルを {filepath} に保存しました")
