# app/predictor.py
import joblib
from pathlib import Path
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Load model and preprocessor once
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
model = joblib.load(MODELS_DIR / "model.joblib")

def predict_heart_disease(input_df: pd.DataFrame):
    """
    Takes a DataFrame of inputs and returns predictions & probabilities
    """
    X_processed = preprocessor.transform(input_df)
    preds = model.predict(X_processed)
    probs = model.predict_proba(X_processed)[:,1]  # probability of class 1
    input_df["predicted_heart_disease"] = preds
    input_df["prediction_prob"] = probs
    return input_df
