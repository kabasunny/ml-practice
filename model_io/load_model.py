import joblib
import os


# モデルの読み込み
def load_model(model_type, directory="trained_models"):
    filename = f"trained_model_{model_type}.pkl"
    filepath = os.path.join(directory, filename)
    model = joblib.load(filepath)
    print(f"モデルを {filepath} から読み込みました")
    return model
