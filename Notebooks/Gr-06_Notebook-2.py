#!/usr/bin/env python
# coding: utf-8

# 
# ## Model Deployment Using FastAPI
# 
# This section demonstrates how to deploy the trained ML model as a RESTful API using FastAPI.
# The API loads the best model from the MLflow artifact store and provides an endpoint (`/predict`)
# that accepts input features in JSON format and returns a prediction.

# In[16]:


from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import mlflow.sklearn
import pandas as pd
import uvicorn
import os

# Define the input schema exactly as per the raw dataset.
class PredictionInput(BaseModel):
    holiday: Optional[str] = None
    temp: float
    rain_1h: float
    snow_1h: float
    clouds_all: int
    weather_main: str
    weather_description: str
    date_time: str  # Format: "YYYY-MM-DD HH:MM:SS"

# Create FastAPI app.
app = FastAPI()

# Load the best model from MLflow.
# Update the model_uri with your actual model's artifact path or MLflow URI.
model_uri = "C:/Users/india/Desktop/Jio_Institute/MLOps/Project/Nagendra/MLOPS/mlruns/676113837923995950/99140a2385dd4c6aa5c0b8ec80c133ff/artifacts/model"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert the input JSON to a Pandas DataFrame (one row)
    input_df = pd.DataFrame([input_data.dict()])
    
    # The model pipeline should include any preprocessing to convert raw inputs
    # into the transformed features used for prediction.
    prediction = model.predict(input_df)
    
    # Return the prediction as a JSON-serializable list.
    return {"predictions": prediction.tolist()}




