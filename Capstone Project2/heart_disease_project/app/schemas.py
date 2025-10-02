# app/schemas.py
from pydantic import BaseModel

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_bp: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_induced_angina: int
    st_depression: float
    st_slope: int
    num_major_vessels: int
    thalassemia: int
