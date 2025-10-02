from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import predict_heart_disease

app = FastAPI()

class HeartRequest(BaseModel):
    age: float
    sex: str
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: float
    max_heart_rate: float
    oldpeak: float
    chest_pain_type: str
    resting_ecg: str
    exercise_induced_angina: str
    st_slope: str

class HeartResponse(BaseModel):
    prediction: int
    message: str

@app.post("/predict", response_model=HeartResponse)
def predict(data: HeartRequest):
    input_data = data.dict()
    pred = predict_heart_disease(input_data)
    message = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    return HeartResponse(prediction=pred, message=message)
