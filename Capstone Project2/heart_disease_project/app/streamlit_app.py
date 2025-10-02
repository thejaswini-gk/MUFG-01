# app/streamlit_app.py
import streamlit as st
import pandas as pd
from predictor import predict_heart_disease

st.title("Heart Disease Prediction")

st.write("Enter patient details below:")

# Create inputs
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
max_heart_rate = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
exercise_induced_angina = st.selectbox("Exercise Induced Angina", [0, 1])
st_slope = st.selectbox("ST Slope", [0, 1, 2])

# Collect into DataFrame
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "chest_pain_type": chest_pain_type,
    "resting_blood_pressure": resting_blood_pressure,
    "cholesterol": cholesterol,
    "max_heart_rate": max_heart_rate,
    "oldpeak": oldpeak,
    "fasting_blood_sugar": fasting_blood_sugar,
    "resting_ecg": resting_ecg,
    "exercise_induced_angina": exercise_induced_angina,
    "st_slope": st_slope
}])

if st.button("Predict"):
    result_df = predict_heart_disease(input_df)
    
    # Map numeric prediction to human-readable text
    pred_label = "Heart Disease Predicted" if result_df["predicted_heart_disease"].iloc[0] == 1 else "No Heart Disease"
    
    st.write("Prediction:", pred_label)
    st.write("Probability:", f"{float(result_df['prediction_prob'].iloc[0])*100:.2f}%")
