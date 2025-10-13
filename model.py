"""
Handles the heart disease prediction model logic, including data preprocessing,
prediction, and performance evaluation.
"""

# --- Core Imports ---
import joblib
import pandas as pd
from typing import List, Dict, Any, Tuple

class HeartDiseaseModel:
    """A class to load the model and artifacts, and make predictions."""

    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        """Initializes the model by loading necessary artifacts."""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            print("Model, scaler, and feature names loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading model artifacts: {e}")
            self.model = None
            self.scaler = None
            self.feature_names = None
        pass

    def _preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transforms raw input data into a scaled DataFrame ready for prediction.
        """
        # TODO: Implement the data preprocessing steps.
        # 1. Convert the input `data` (a list of dictionaries) into a pandas DataFrame.
        # 2. Use pd.get_dummies() to one-hot encode categorical features.
        # 3. Use reindex() to ensure it has the same columns as `self.feature_names` and fill any missing columns with 0.
        # 4. Use `self.scaler.transform` to transform the aligned data.
        # 5. Return the resulting scaled features.
        pass

    def predict(self, data: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
        """
        Generates predictions and probabilities for the input data.
        """
        # TODO: Implement the prediction logic.
        # 1. Call `self._preprocess_data()` with the input `data` to get the scaled features.
        # 2. Use `self.model.predict()` on the scaled features to get predictions.
        # 3. Use `self.model.predict_proba()` to get the prediction probabilities for the positive class (class 1).
        # 4. Convert both predictions and probabilities to lists and return them as a tuple.
        pass


# Instantiate the model once when the module is imported.
# This makes it a singleton that can be imported by the server.
model_pipeline = HeartDiseaseModel(
    model_path="model/heart_disease_classifier.pkl",
    scaler_path="model/data_scaler.joblib",
    features_path="model/feature_names.joblib"
)