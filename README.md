# mlops-workshop-2025

## Project Assumptions

For this MLOps workshop, we are making several simplifying assumptions to focus on the core concepts of deploying a model.

* **Data is Ready**: We assume the `heart_disease.csv` dataset is already cleaned, trustworthy, and ready for model training. We are skipping the in-depth exploratory data analysis (EDA) and feature engineering steps.
* **Model is Chosen**: We assume that a `RandomForestClassifier` is a "good enough" model for this problem. We are not focusing on hyperparameter tuning or comparing the performance of different algorithms. The main goal is to operationalize a working model, not perfect it.
* **Local Development Environment**: We assume the entire workflow—from training to serving the API—will run on a local machine. We are not dealing with cloud deployment, containerization (like Docker), or CI/CD pipelines in this initial stage.
* **Problem is Defined**: We assume the business problem ("predict heart disease") and the primary metric for success (accuracy) are already defined and agreed upon.

---
## Project Overview for the Workshop

### The Goal
Our primary goal is to bridge the gap between a trained machine learning model and a real-world application. A model saved as a file on a data scientist's laptop is not useful; we need to make it accessible, reliable, and easy for other services to use. This is the core challenge of MLOps.

### The Model
We are working with a **Random Forest Classifier** that has been trained to predict a binary outcome: whether a patient has heart disease (`1`) or not (`0`). The model uses various patient health metrics as input features, such as age, sex, cholesterol levels, and chest pain type.

### Our MLOps Workflow
To make our model useful, we will implement a three-part system that represents a foundational MLOps pipeline.

1.  **The Training Pipeline (`train.py`)**
    This script is our **reproducible factory for the model**. Its job is to take the raw dataset and produce all the necessary components, or "artifacts," for prediction.
    * It loads and preprocesses the data.
    * It trains the Random Forest model.
    * It saves the three key artifacts: the trained model (`.pkl`), the data scaler (`.joblib`), and the list of feature names (`.joblib`).

2.  **The Model Wrapper (`model.py`)**
    This script acts as a **clean interface to our model**. It abstracts away the complexity of preprocessing and prediction.
    * It defines a `HeartDiseaseModel` class that loads the three artifacts created by the training script.
    * It handles the logic for taking new, raw data (in JSON format), applying the correct preprocessing and scaling, and calling the model's `.predict()` method.
    * It exposes a single, easy-to-use `model_pipeline` object for our server.

3.  **The API Server (`server.py`)**
    This is the **front door to our model**. It uses the FastAPI framework to expose our model's functionality to the outside world over the internet.
    * It defines the API endpoints (like `/predict`).
    * It uses Pydantic models to validate incoming data, ensuring we don't get errors.
    * When a request comes in, it passes the validated data to our `model_pipeline` object and returns the prediction in a clean JSON format.
