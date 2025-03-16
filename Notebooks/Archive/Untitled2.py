#!/usr/bin/env python
# coding: utf-8

# In[41]:


from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import mlflow.sklearn
import pandas as pd
import uvicorn

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
model_uri = "C:/Users/india/Desktop/Jio_Institute/MLOps/Project/Nagendra/MLOPS/Notebooks/mlruns/590277716352978090/57436f52a1df43439a9a1d4bbff2e0b8/artifacts/model"
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





