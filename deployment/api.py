from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd

# --------------------------
# Load saved artifacts
# --------------------------
model = xgb.Booster()
model.load_model("models/best_model_gwo.json")

scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(
    title="AML XGBoost + Swarm Optimization API",
    description="Predict money laundering probability using optimized XGBoost",
    version="1.0",
)

# --------------------------
# Input schema
# --------------------------
class Transaction(BaseModel):
    Timestamp: int
    Payment_Format: int
    Amount: float
    OldBalanceOrig: float
    NewBalanceOrig: float
    OldBalanceDest: float
    NewBalanceDest: float


# --------------------------
# Helper function
# --------------------------
def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    df = df[feature_columns]                     # correct order
    df_scaled = scaler.transform(df)             # normalize
    dmatrix = xgb.DMatrix(df_scaled)
    return dmatrix


# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    data_dict = transaction.dict()

    dmatrix = preprocess_input(data_dict)

    prob = float(model.predict(dmatrix)[0])      # probability
    label = int(prob >= 0.5)                     # classification

    return {
        "prediction": label,
        "probability": round(prob, 5)
    }


@app.get("/")
def root():
    return {"message": "AML XGBoost API is running"}
