#!/usr/bin/env python
# coding: utf-8

# # MLOPS Project Group 06
# **Team: Nagendra Jupudy, Vamsi Krishna Pirati, Rajesh Avunoori, Sanju Vikasini Velmurugan, Piyush Borse**
# ## **Predicting Metro_Interstate_Traffic_Volume**
# 
# **Dataset Overview:**
# - **Domain:** Transportation / Traffic Analysis  
# - **Task:** Regression   
# - **Dataset Type:** Multivariate  
# - **Number of Instances:** 48,204
# - **Number of Features:** 8  
# - **Feature Types:** Mixed (categorical, continuous, integer) 
# 
# **Objective:**  
# The primary objective of this project is to build an effective regression model to forecast traffic volume on the Metro Interstate  

# In[24]:


import sys
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandera as pa_schema
from pandera import Column, DataFrameSchema, Check
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check


# In[26]:


import pandas as pd
import pyarrow as pa  # Ensure this is the pyarrow library
import numpy as np


# # Environment and Working Directory Info
# 

# In[29]:


print(sys.executable)


# In[31]:


import sys
print(sys.executable)


# In[33]:


get_ipython().system('conda info --envs')
get_ipython().system('where jupyter')


# In[34]:


# change directory if required
get_ipython().run_line_magic('cd', 'C:\\Users\\india\\Desktop\\Jio_Institute\\MLOps\\Project\\Nagendra\\MLOPS')


# In[37]:


print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir(os.getcwd()))


# In[39]:


# Define path to the original CSV file
csv_file_path = os.path.join("Dataset", "Original", "Metro_Interstate_Traffic_Volume.csv")

# Read the CSV file
data = pd.read_csv(csv_file_path, sep=",", engine="python")
print(data.head())


# ## Data Loading and Initial Inspection
# 
# In this section, we load the dataset from a CSV file using pandas and inspect the first few rows. This initial step helps to verify that the data has been imported correctly and provides a quick look at the dataset's structure.

# In[42]:


# # Define the schema (data types) for each column
# dtype_schema = {
#     "holiday": "object",
#     "temp": "float64",
#     "rain_1h": "float64",
#     "snow_1h": "float64",
#     "clouds_all": "int64",
#     "weather_main": "object",
#     "weather_description": "object",
#     "traffic_volume": "int64"
# }


# In[44]:


# Define the PyArrow schema based on the provided dtype schema
schema = pa.schema([
    pa.field("holiday", pa.string()),
    pa.field("temp", pa.float64()),
    pa.field("rain_1h", pa.float64()),
    pa.field("snow_1h", pa.float64()),
    pa.field("clouds_all", pa.int64()),
    pa.field("weather_main", pa.string()),
    pa.field("weather_description", pa.string()),
    pa.field("traffic_volume", pa.int64())
])

print(schema)


# In[48]:


# Read the CSV file using the defined schema and parse 'date_time' as datetime
df = pd.read_csv(
    os.path.join("Dataset", "Original", "Metro_Interstate_Traffic_Volume.csv"),
    dtype=dtype_schema,
    parse_dates=["date_time"]
)


# In[50]:


df.info()


# In[ ]:





# # Export DataFrame to Parquet

# In[53]:


# Define output directory for Parquet files and create it if necessary
output_dir = os.path.join("Dataset", "Parquet")
os.makedirs(output_dir, exist_ok=True)


# In[55]:


# Convert the DataFrame to a PyArrow Table
table = pa.Table.from_pandas(df)

# Write the table to a Parquet file
parquet_file_path = os.path.join(
    output_dir, "Metro_Interstate_Traffic_Volume.parquet"
)
pq.write_table(table, parquet_file_path)


# # Data Profiling using ydata_profiling

# In[57]:


# Generate the profile report
profile = ProfileReport(
    df,
    title="Metro Interstate Traffic Volume Data Profiling Report",
    explorative=True
)

# Save the profiling report as an HTML file
profile.to_file("Metro_Interstate_Traffic_Volume_Profile.html")

# Display the profiling report in a Jupyter Notebook iframe
profile.to_notebook_iframe()


# In[56]:


df_parquet = pd.read_parquet("Dataset/Parquet/Metro_Interstate_Traffic_Volume.parquet")


# # Split Data into Train (60%), Test (20%), and Prod (20%) Sets

# In[58]:


# Ensure the DataFrame is sorted by date_time for time series splitting
df_parquet = df_parquet.sort_values(by="date_time")

# Get the total number of rows
n_total = len(df_parquet)

# Calculate split indices for 60%, 20%, and 20%
train_end = int(0.6 * n_total)
test_end = int(0.8 * n_total)

# Split the data into train, test, and production sets
train_df = df_parquet.iloc[:train_end]
test_df = df_parquet.iloc[train_end:test_end]
prod_df = df_parquet.iloc[test_end:]

print("Metro_Interstate_Traffic_Volume_train:", len(train_df))
print("Metro_Interstate_Traffic_Volume_test:", len(test_df))
print("Metro_Interstate_Traffic_Volume_prod:", len(prod_df))

# Save each split back to a Parquet file
train_df.to_parquet(
    os.path.join(output_dir, "Metro_Interstate_Traffic_Volume_train.parquet"),
    index=False
)
test_df.to_parquet(
    os.path.join(output_dir, "Metro_Interstate_Traffic_Volume_test.parquet"),
    index=False
)
prod_df.to_parquet(
    os.path.join(output_dir, "Metro_Interstate_Traffic_Volume_prod.parquet"),
    index=False
)


# In[ ]:


# !git init
# !git status
# !git add .
# !git status
# !git commit -m "Initial Version 2: updated in PEP8 standard"
# !git push


# In[91]:


unique_holidays = df['weather_description'].unique()


# In[93]:


len(unique_holidays)


# In[95]:


unique_holidays


# In[ ]:




