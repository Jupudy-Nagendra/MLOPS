# MLOps Project: Metro Interstate Traffic Volume Prediction

This project implements an end-to-end MLOps pipeline for forecasting traffic volume on the Metro Interstate. The pipeline includes data ingestion, validation, experiment tracking, model training, deployment, user interface development, and model monitoring.

---

## Project Overview

- **Data Preparation & Validation:**  
  The dataset is ingested from Parquet files, validated, and split into training, test, and production sets. Feature engineering includes extracting datetime components (year, month, day, hour) and transforming features such as holiday and weather details.

- **Experiment Tracking & Model Training:**  
  Multiple regression models (e.g., Linear Regression, Ridge, Random Forest, XGBoost, etc.) are evaluated using cross-validation and test metrics. All experiments are tracked with MLflow to identify the best-performing model based on error metrics like MAE, RMSE, and R².

- **Model Deployment:**  
  The selected model is deployed using FastAPI, providing a RESTful API endpoint that serves real-time predictions.

- **User Interface Development:**  
  A Streamlit UI enables users to input data for all features and receive predictions from the FastAPI endpoint, ensuring an interactive experience.

- **Model Monitoring & Data Drift Analysis:**  
  Data drift is monitored using alibi-detect for numeric features and chi-square tests for categorical features. This analysis ensures that the production data remains consistent with the training data, allowing proactive model retraining when needed.

---

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/<username>/MLOps_MetroTrafficPrediction.git
cd MLOps_MetroTrafficPrediction
```
## Running the Notebooks and Components

### Notebook 1: Data Preparation, Validation, and Experiment Tracking
- **Content:**  
  Contains code for data ingestion, validation with Pandera, data profiling, splitting the dataset into training, test, and production sets, building the ML pipeline for traffic volume prediction, and tracking experiments with MLflow.
- **Usage:**  
  Open and run `Notebook.ipynb` in your Jupyter environment to reproduce the experiments.

### Notebook 2: Model Deployment using FastAPI
- **Content:**  
  Contains FastAPI code to deploy the best regression model for forecasting traffic volume.
- **Converted to Python File:**  
  The code has been converted to `Notebook2.py` for command-line execution.
- **Run the FastAPI App:**
  ```bash
  cd MLOPS Project/notebooks
  uvicorn Gr-06_Notebook-2:app --host 127.0.0.1 --port 8000 --reload
  ```
### Notebook 3: User Interface with Streamlit
- **Content:**  
  Contains the Streamlit UI code for collecting user inputs (such as holiday, temperature, rainfall, snowfall, cloud coverage, weather conditions, and date/time) and displaying traffic volume predictions from the deployed FastAPI endpoint.
- **Converted to Python File:**  
  The code has been converted to `Notebook3.py` for command-line execution.
- **Run the Streamlit UI:**
  ```bash
  cd MLOPS_Project/notebooks
  streamlit run Gr-06_Notebook-3.py
     ```
### Notebook 4: Model Monitoring (Data Drift Detection)
- **Content:**  
  Uses alibi-detect for numeric drift detection and chi-square tests for categorical drift detection to monitor changes in feature distributions between the training and production datasets.
- **Usage:**  
  Run `Notebook4.ipynb` in your Jupyter environment to view drift detection results in tabular format.

