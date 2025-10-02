import streamlit as st
import pandas as pd
import joblib

# Load model and features
model, features = joblib.load("model.pkl")

st.title("🏭 Manufacturing Output Prediction")
st.write("Enter machine parameters to predict Parts Per Hour:")

# Collect user input for all features used in training
input_data = {}
for f in features:
    input_data[f] = st.number_input(f, value=0.0)

# Predict
if st.button("Predict"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.success(f"✅ Predicted Parts Per Hour: {prediction:.2f}")
