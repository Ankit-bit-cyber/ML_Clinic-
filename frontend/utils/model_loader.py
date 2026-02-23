import joblib
import os

MODEL_PATH = os.path.join("artifacts", "model.pkl")

def load_model():
    return joblib.load(MODEL_PATH)