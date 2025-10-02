# src/train.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import joblib

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_disease_dataset.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.rename(columns={'st_depression': 'oldpeak'}, inplace=True)  # optional rename
    return df

def build_preprocessor(df):
    expected_numeric = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'oldpeak']
    expected_categorical = ['sex','chest_pain_type','fasting_blood_sugar','resting_ecg','exercise_induced_angina','st_slope']

    numeric_cols = [c for c in expected_numeric if c in df.columns]
    categorical_cols = [c for c in expected_categorical if c in df.columns]

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols)
    ], remainder="drop")

    return preprocessor

def main():
    df = load_data()
    X = df.drop(columns=["heart_disease"])
    y = df["heart_disease"]

    preprocessor = build_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)

    pipeline = Pipeline([("clf", RandomForestClassifier(random_state=42))])

    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_leaf": [1, 2]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_processed, y)

    best_model = grid.best_estimator_

    # Save model & preprocessor separately
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    print("Saved preprocessor and model to models/")

if __name__ == "__main__":
    main()
