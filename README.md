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
  

    Of course. Here is a comprehensive `README.md` file for your project.

-----

### Key Features

  - **REST API**: Exposes the prediction model through a simple and robust API.
  - **Data Validation**: Uses Pydantic to ensure all incoming data is in the correct format.
  - **Model Training Pipeline**: A repeatable script (`train.py`) to preprocess data, train a classifier, and save the necessary artifacts.
  - **Efficient**: The model and scaler are loaded only once when the server starts, ensuring low-latency predictions.
  - **Interactive Documentation**: FastAPI automatically generates interactive API documentation (using Swagger UI).

-----

## Project Structure

```
heart-disease-api/
├── model/
│   ├── heart_disease_classifier.pkl  (created after training)
│   ├── data_scaler.joblib          (created after training)
│   └── feature_names.joblib        (created after training)
├── heart_disease.csv
├── model.py
├── server.py
├── train.py
└── requirements.txt
```

-----

## Setup and Installation

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd heart-disease-api
```

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
fastapi==0.103.2
uvicorn[standard]==0.23.2
scikit-learn==1.3.1
pandas==2.1.1
joblib==1.3.2
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 4\. Dataset

Make sure the `heart_disease.csv` dataset is in the root directory of the project.

-----

## Usage

### Step 1: Train the Model

Run the training script to process the data and save the model artifacts. This will create the `model/` directory and its contents.

```bash
python train.py
```

### Step 2: Run the API Server

Start the API server using Uvicorn.

```bash
uvicorn server:app --reload
```

The server will be running at `http://127.0.0.1:8000`. The `--reload` flag automatically restarts the server when you make changes to the code.

### Step 3: Access the Interactive Docs

Once the server is running, navigate to **`http://127.0.0.1:8000/docs`** in your browser to see the interactive API documentation. You can test the endpoints directly from this interface.

-----

## API Endpoints

### Root

  - **URL**: `/`
  - **Method**: `GET`
  - **Description**: A simple health check endpoint to confirm the API is running.
  - **Success Response**:
    ```json
    {
      "status": "ok",
      "message": "Welcome to the Heart Disease Prediction API!"
    }
    ```

### Predict

  - **URL**: `/predict`
  - **Method**: `POST`
  - **Description**: Predicts heart disease for a batch of one or more patients.
  - **Request Body**:
    ```json
    {
      "data": [
        {
          "Age": 58,
          "Sex": "M",
          "ChestPainType": "ATA",
          "RestingBP": 120,
          "Cholesterol": 284,
          "FastingBS": 0
        },
        {
          "Age": 42,
          "Sex": "F",
          "ChestPainType": "NAP",
          "RestingBP": 130,
          "Cholesterol": 200,
          "FastingBS": 1
        }
      ]
    }
    ```
  - **Success Response**:
    ```json
    {
      "predictions": [0, 1],
      "probabilities": [0.12, 0.88]
    }
    ```
