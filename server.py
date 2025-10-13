# server.py
"""
FastAPI application server for heart disease prediction.

This module contains the main FastAPI application, including API endpoint
definitions that call the prediction model.
To run this server, use the command: `uvicorn server:app --reload`
"""

# --- Core Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any, Optional

# --- Local Imports ---
# Import the model pipeline instance from our model module
from model import model_pipeline

# --- Application Setup ---
app = FastAPI(
    title="Heart Disease Prediction API",
    description="An API to predict heart disease from patient data and calculate model performance.",
    version="1.0.0"
)

# --- Pydantic Models for Data Validation ---
class PredictionInput(BaseModel):
    """Defines the structure for a batch prediction request."""
    data: List[Dict[str, Any]]
    true_labels: Optional[List[int]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"Age": 58, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 284, "FastingBS": 0},
                    {"Age": 42, "Sex": "F", "ChestPainType": "NAP", "RestingBP": 130, "Cholesterol": 200, "FastingBS": 1}
                ],
                "true_labels": [0, 1]
            }
        }

class PredictionResponse(BaseModel):
    """Defines the structure for the prediction response."""
    predictions: List[int]
    probabilities: List[float]
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None

# --- API Endpoints ---

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Heart Disease Prediction API!"}


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(payload: PredictionInput):
    """
    Processes a batch of patient data to predict heart disease and evaluates performance.
    """
    if not model_pipeline.model:
        raise HTTPException(status_code=503, detail="Model not loaded. The service is unavailable.")

    try:
        response = None
        #TODO: Implement the prediction endpoint logic.
        # Get predictions and probabilities from the model pipeline
        # 1. Call `model_pipeline.predict()` with `payload.data` to get predictions and probabilities.
        # 2. If `payload.true_labels` is provided, call `model_pipeline.evaluate()` to get accuracy and F1 score.
        # 3. Return a `PredictionResponse` containing predictions, probabilities, and optionally accuracy and F1 score.
        return response
    
    except RuntimeError as e:
         raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)