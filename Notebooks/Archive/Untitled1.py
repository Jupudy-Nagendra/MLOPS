#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------------------
# Step 1: Define raw URLs for train, test, and prod datasets.
# -------------------------------
train_url = "https://raw.githubusercontent.com/Jupudy-Nagendra/MLOPS/main/Dataset/Parquet/Metro_Interstate_Traffic_Volume_train.parquet"
test_url  = "https://raw.githubusercontent.com/Jupudy-Nagendra/MLOPS/main/Dataset/Parquet/Metro_Interstate_Traffic_Volume_test.parquet"
prod_url  = "https://raw.githubusercontent.com/Jupudy-Nagendra/MLOPS/main/Dataset/Parquet/Metro_Interstate_Traffic_Volume_prod.parquet"

# Helper function to load a parquet file from GitHub.
def load_parquet_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_parquet(BytesIO(response.content))

# Load the train dataset (for example).
train_df = load_parquet_from_url(train_url)
test_df  = load_parquet_from_url(test_url)
prod_df  = load_parquet_from_url(prod_url)

# -------------------------------
# Step 2: Define a custom transformer to extract datetime features
# -------------------------------
class DateTimeExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that converts the 'date_time' column into datetime format
    and extracts new columns: year, month, day, and hour.
    Optionally drops the original date_time column.
    """
    def __init__(self, column="date_time", drop_original=True):
        self.column = column
        self.drop_original = drop_original
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert the column to datetime
        X[self.column] = pd.to_datetime(X[self.column])
        # Extract datetime components
        X['year'] = X[self.column].dt.year
        X['month'] = X[self.column].dt.month
        X['day'] = X[self.column].dt.day
        X['hour'] = X[self.column].dt.hour
        # Optionally drop the original date_time column
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X

# Build the datetime extraction pipeline.
datetime_pipeline = Pipeline(steps=[
    ('datetime_extractor', DateTimeExtractor(column="date_time", drop_original=True))
])

# -------------------------------
# Step 3: Define a custom holiday transformer
# -------------------------------
class HolidayBinaryTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts the 'holiday' column into a binary feature:
      - 0 if the value is missing or equals "None"
      - 1 otherwise.
    Implements get_feature_names_out for compatibility.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expect X as a DataFrame with column 'holiday'
        binary = ((~pd.isnull(X)) & (X != "None")).astype(int)
        # Ensure a 2D array output
        if isinstance(binary, pd.Series):
            binary = binary.to_frame()
        return binary.values

    def get_feature_names_out(self, input_features=None):
        return np.array(['holiday_binary'])

# Instantiate our custom holiday transformer.
holiday_transformer = HolidayBinaryTransformer()

# -------------------------------
# Step 4: Define pipelines for numeric and categorical features
# -------------------------------
# After datetime extraction, the DataFrame will include:
# Numeric columns: 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'year', 'month', 'day', 'hour'
# Categorical columns (excluding 'holiday'): 'weather_main', 'weather_description'
numeric_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'year', 'month', 'day', 'hour']
categorical_cols = ['weather_main', 'weather_description']

# Numeric pipeline: impute missing values and scale.
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values and one-hot encode.
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# -------------------------------
# Step 5: Create a ColumnTransformer to combine all preprocessing steps
# -------------------------------
# The preprocessor applies:
# - Numeric processing on numeric_cols.
# - Categorical processing on categorical_cols.
# - The holiday transformer on the 'holiday' column.
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols),
    ('holiday', holiday_transformer, ['holiday'])
])

# -------------------------------
# Step 6: Build the full pipeline
# -------------------------------
# Chain datetime extraction and the preprocessor.
full_pipeline = Pipeline(steps=[
    ('datetime', datetime_pipeline),   # Extract datetime features
    ('preprocessor', preprocessor)       # Process numeric, categorical, and holiday columns
    # A model step can be added here, e.g., ('model', SomeRegressor())
])

# -------------------------------
# Step 7: Prepare feature data and transform
# -------------------------------
# Remove the target variable ('traffic_volume') from the train dataset.
X_train_features = train_df.drop(columns=['traffic_volume'])

# Fit and transform the features using the full pipeline.
X_transformed = full_pipeline.fit_transform(X_train_features)

# If OneHotEncoder returns a sparse matrix, convert it to a dense array.
X_transformed_dense = X_transformed if isinstance(X_transformed, np.ndarray) else X_transformed.toarray()

# -------------------------------
# Step 8: Retrieve Output Feature Names
# -------------------------------
# Attempt to get feature names from the preprocessor step.
try:
    feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
    print("Output Feature Names:")
    print(feature_names)
except Exception as e:
    print("Error retrieving feature names:", e)

# -------------------------------
# Step 9: Inspect Transformed Data
# -------------------------------
print("\nTransformed data shape:", X_transformed_dense.shape)
print("First row of transformed data:")
print(X_transformed_dense[0])


# In[15]:


import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb


# In[11]:


target_col = 'traffic_volume'

# For train data
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# For test data
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# For production data (no target provided, for inference)
X_prod = prod_df.copy()

# ---------------------------------------
# 4. Transform features using the full pipeline
# ---------------------------------------
X_train_transformed = full_pipeline.fit_transform(X_train)
X_test_transformed  = full_pipeline.transform(X_test)
X_prod_transformed  = full_pipeline.transform(X_prod)

# ---------------------------------------
# 5. Train an XGBoost Regressor
# ---------------------------------------
# Create and train an XGBoost regressor on the preprocessed train data.
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
model.fit(X_train_transformed, y_train)

# ---------------------------------------
# 6. Evaluate the Model on Test Data
# ---------------------------------------
y_pred_test = model.predict(X_test_transformed)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
print("Test MAPE: {:.2%}".format(mape_test))

# ---------------------------------------
# 7. Predict on Production Data
# ---------------------------------------
prod_predictions = model.predict(X_prod_transformed)
print("Production predictions (first 5):")
print(prod_predictions[:5])


# In[22]:


from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# Create the full pipeline by combining the preprocessor and the XGBoost regressor
xgb_pipeline = Pipeline(steps=[('datetime', datetime_pipeline),
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42))
])

# Fit the pipeline on the training data (X_train are the features and y_train is the target)
xgb_pipeline.fit(X_train, y_train)



# In[17]:


# Compute error metrics.
mae   = mean_absolute_error(y_test, y_pred_test)
mse   = mean_squared_error(y_test, y_pred_test)
rmse  = np.sqrt(mse)
mape  = mean_absolute_percentage_error(y_test, y_pred_test)
r2    = r2_score(y_test, y_pred_test)

print("Error Metrics on Test Data:")
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("Mean Absolute Percentage Error (MAPE): {:.2%}".format(mape))
print("R² Score: {:.2f}".format(r2))


# In[28]:


get_ipython().system('pip install mlflow')


# In[30]:


import mlflow
import mlflow.sklearn


# In[67]:


import os
print("Current working directory:", os.getcwd())


# In[92]:


get_ipython().run_line_magic('cd', 'C:\\Users\\india\\Desktop\\Jio_Institute\\MLOps\\Project\\Nagendra\\MLOPS')


# In[94]:


ls


# In[72]:


# Set up MLflow experiment (all runs will be grouped under this experiment)
mlflow.set_experiment("Metro_Interstate_Traffic_Volume")


# In[74]:


import mlflow

print("Current Tracking URI:", mlflow.get_tracking_uri())


# In[76]:


import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")


# In[78]:


regression_models = {
    "LinearRegression_default": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),

}


# In[80]:


for model_name, reg_model in regression_models.items():
    with mlflow.start_run(run_name=model_name):
        # Log the model name as a parameter.
        mlflow.log_param("model_name", model_name)
        
        # Create a full pipeline: preprocessing + regressor.
        pipeline = Pipeline(steps=[('datetime', datetime_pipeline),
            ('preprocessor', preprocessor),
            ('regressor', reg_model)
        ])
        # Define 5-fold cross-validation.
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform 5-fold cross-validation using negative MAE (we will take the absolute value).
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
        mean_cv = np.mean(np.abs(cv_scores))
        std_cv = np.std(np.abs(cv_scores))
        
        # Log cross-validation metrics.
        mlflow.log_metric("cv_mean_MAE", mean_cv)
        mlflow.log_metric("cv_std_MAE", std_cv)
        
        # Train the pipeline on the full training data.
        pipeline.fit(X_train, y_train)
        
        # Evaluate on the test set.
        y_pred = pipeline.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        
        # Log test metrics.
        mlflow.log_metric("test_MAE", mae)
        mlflow.log_metric("test_MSE", mse)
        mlflow.log_metric("test_RMSE", rmse)
        mlflow.log_metric("test_MAPE", mape)
        mlflow.log_metric("test_R2", r2)
        
        # Log hyperparameters if applicable.
        if model_name.startswith("LinearRegression") or model_name.startswith("Ridge") or model_name.startswith("Lasso") or model_name.startswith("ElasticNet"):
            # For these models, log the regularization strength if available.
            if hasattr(reg_model, "alpha"):
                mlflow.log_param("alpha", reg_model.alpha)
            if hasattr(reg_model, "l1_ratio"):
                mlflow.log_param("l1_ratio", reg_model.l1_ratio)
        
        # Log the trained model pipeline.
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"{model_name} -> CV MAE: {mean_cv:.4f} ± {std_cv:.4f}, Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")


# In[1]:


ls


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Untitled1.ipynb')


# In[ ]:




