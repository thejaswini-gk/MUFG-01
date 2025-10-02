import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("manufacturing_dataset_1000_samples.csv")

# Strip spaces in column names
df.columns = df.columns.str.strip()

# Target column
target = "Parts_Per_Hour"

# Keep only numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Drop rows without target
numeric_df = numeric_df.dropna(subset=[target])

# Fill missing values in features with median
for col in numeric_df.columns:
    if col != target:
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

X = numeric_df.drop(columns=[target])
y = numeric_df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained successfully")
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model and feature names
joblib.dump((model, list(X.columns)), "model.pkl")
print("✅ Model saved as model.pkl")
