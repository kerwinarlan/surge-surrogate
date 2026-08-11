import os
import time

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SurgeSurrogate API", description="Fast ML Hydrodynamic Surrogate")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "surge_model.pkl")
# Model feature order must match pipeline.MODEL_FEATURES.
FEATURES = [
    "wind_kts",
    "pressure_deficit_hpa",
    "distance_to_manila_km",
    "approach_angle_deg",
]

_model = None


def load_model():
    """Load the trained XGBoost model once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class StormInput(BaseModel):
    wind_kts: float
    pressure_deficit_hpa: float
    distance_to_manila_km: float
    approach_angle_deg: float


@app.post("/predict")
async def predict_surge(storm: StormInput):
    model = load_model()
    start = time.perf_counter()

    X = np.array(
        [[
            storm.wind_kts,
            storm.pressure_deficit_hpa,
            storm.distance_to_manila_km,
            storm.approach_angle_deg,
        ]]
    )
    predicted_surge_residual_m = float(model.predict(X)[0])

    inference_time_ms = (time.perf_counter() - start) * 1000

    return {
        "status": "success",
        "predicted_surge_residual_meters": round(predicted_surge_residual_m, 2),
        "inference_time_ms": round(inference_time_ms, 2),
    }


@app.get("/")
async def root():
    return {"message": "SurgeSurrogate API is online. Visit /docs for the interactive API runner."}
