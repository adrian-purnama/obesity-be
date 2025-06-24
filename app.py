# app.py
import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np

# Load model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define request schema
class ObesityInput(BaseModel):
    Age: int
    Height: float
    Weight: float
    FCVC: float
    NCP: float
    CH2O: float
    FAF: float
    TUE: float
    Gender: str
    family_history_with_overweight: str
    FAVC: str
    CAEC: str
    SMOKE: str
    SCC: str
    CALC: str
    MTRANS: str

@app.post("/predict")
def predict(data: ObesityInput):
    # Convert input to list of features — must match training preprocessing
    input_data = [[
        data.Age, data.Height, data.Weight, data.FCVC, data.NCP, data.CH2O,
        data.FAF, data.TUE, data.Gender, data.family_history_with_overweight,
        data.FAVC, data.CAEC, data.SMOKE, data.SCC, data.CALC, data.MTRANS
    ]]
    
    # Preprocessing must match training! This is just a placeholder.
    # You need to load and apply your preprocessor pipeline here.

    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
