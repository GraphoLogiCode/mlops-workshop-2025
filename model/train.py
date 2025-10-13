"""
Model training script for the Heart Disease Prediction API.

This script performs the following steps:
1.  Loads the heart disease dataset from `heart.csv`.
2.  Preprocesses the data by applying one-hot encoding to categorical features.
3.  Splits the data into training and testing sets.
4.  Applies standard scaling to the feature set.
5.  Trains a RandomForestClassifier model.
6.  Evaluates the model's performance on the test set.
7.  Saves the trained model, the fitted scaler, and the feature names to the 'model/'
    directory for use by the FastAPI application.
"""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Data Loading ---
print("Loading dataset...")
try:
    df = pd.read_csv('heart_disease.csv')
except FileNotFoundError:
    print("Error: 'heart_disease.csv' not found. Make sure the dataset is in the same directory.")
    exit()

# --- 2. Data Preprocessing ---
print("Preprocessing data...")
# One-hot encode categorical features
df_processed = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# Separate features (X) and target (y)
X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']

# Store the feature names after encoding, which the API will need
feature_names = X.columns.tolist()

# --- 3. Data Splitting ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Feature Scaling ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on the training data

# --- 5. Model Training ---
print("Training RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 6. Model Evaluation ---
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 7. Saving Artifacts ---
print("Saving model and preprocessing artifacts...")

# Save the trained model
joblib.dump(model, "heart_disease_classifier.pkl")

# Save the fitted scaler
joblib.dump(scaler, "data_scaler.joblib")

# Save the feature names
joblib.dump(feature_names, "feature_names.joblib")

print("\nTraining complete. Artifacts are saved in the 'model/' directory.")
