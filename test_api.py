"""
Script for testing the Heart Disease Prediction API.

This script sends a POST request with sample patient data to the running
FastAPI application and prints the JSON response from the server.

Instructions:
1. Make sure the server is running by executing `uvicorn server:app --reload` in your terminal.
2. Run this script in a separate terminal: `python test_api.py`
"""

import requests
import json

# The URL of the prediction endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Sample data for two patients. The API expects a list of dictionaries.
# We also include the ground truth labels to test the accuracy calculation feature.
test_data = {
    "data": [
        {
            "Age": 58, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 120, 
            "Cholesterol": 284, "FastingBS": 0, "RestingECG": "Normal", 
            "MaxHR": 160, "ExerciseAngina": "N", "Oldpeak": 0, "ST_Slope": "Up"
        },
        {
            "Age": 48, "Sex": "F", "ChestPainType": "ASY", "RestingBP": 138, 
            "Cholesterol": 214, "FastingBS": 0, "RestingECG": "Normal", 
            "MaxHR": 108, "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"
        }
    ]
}

# --- Sending the Request ---
try:
    # Send a POST request to the API with the JSON data
    response = requests.post(API_URL, json=test_data)
    
    # Raise an exception for bad status codes (4xx or 5xx)
    response.raise_for_status() 

    # --- Processing the Response ---
    # Parse the JSON response from the server
    response_data = response.json()
    
    print("--- API Request Sent Successfully ---")
    print(f"Status Code: {response.status_code}")
    print("\n--- API Response Received ---")
    # Use json.dumps for pretty printing the JSON response
    print(json.dumps(response_data, indent=2))

except requests.exceptions.RequestException as e:
    print(f"--- API Request Failed ---")
    print(f"An error occurred: {e}")
