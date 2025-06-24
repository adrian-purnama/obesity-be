from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load pipeline (preprocessor + model) and label encoder
with open("xgb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

app = FastAPI()

# Optional: test endpoint
@app.get("/test")
def test():
    return {"message": "hehe"}

# Define input format (must match training columns exactly)
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
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using full pipeline
    pred_encoded = model.predict(input_df)

    # Decode label
    pred_label = le.inverse_transform(pred_encoded)[0]

    return {"prediction": pred_label}
