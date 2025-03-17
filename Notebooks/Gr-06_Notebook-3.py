#!/usr/bin/env python
# coding: utf-8

# ## User Interface Development with Streamlit
# In this segment, we build an interactive interface for our machine learning application using Streamlit. The interface enables users to enter values for each feature, and when the "Predict" button is pressed, the input is sent to our FastAPI endpoint. The resulting prediction is subsequently displayed on the screen.

# In[1]:


import streamlit as st
import requests
from datetime import datetime, time 

# Title and description
st.title("Traffic Volume Prediction")
st.write("Enter the following information to get a traffic volume prediction:")

# Input fields for each feature from the raw dataset
holiday = st.selectbox("Month of Last Contact", options=["None", 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
       'Christmas Day', 'New Years Day', 'Washingtons Birthday',
       'Memorial Day', 'Independence Day', 'State Fair', 'Labor Day',
       'Martin Luther King Jr Day'])
temp = st.number_input("Temperature (in Kelvin)", min_value=0.0, value=290.0)
rain = st.number_input("Rain in last hour", min_value=0.0, value=0.0)
snow = st.number_input("Snow in last hour", min_value=0.0, value=0.0)
clouds = st.number_input("Clouds All (%)", min_value=0, max_value=100, value=75)
weather_main = st.selectbox("clouds", options=['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog',
       'Thunderstorm', 'Snow', 'Squall', 'Smoke'])
weather_desc = st.selectbox("weather", options=['scattered clouds', 'broken clouds', 'overcast clouds',
       'sky is clear', 'few clouds', 'light rain',
       'light intensity drizzle', 'mist', 'haze', 'fog',
       'proximity shower rain', 'drizzle', 'moderate rain',
       'heavy intensity rain', 'proximity thunderstorm',
       'thunderstorm with light rain', 'proximity thunderstorm with rain',
       'heavy snow', 'heavy intensity drizzle', 'snow',
       'thunderstorm with heavy rain', 'freezing rain', 'shower snow',
       'light rain and snow', 'light intensity shower rain', 'SQUALLS',
       'thunderstorm with rain', 'proximity thunderstorm with drizzle',
       'thunderstorm', 'Sky is Clear', 'very heavy rain',
       'thunderstorm with light drizzle', 'light snow',
       'thunderstorm with drizzle', 'smoke', 'shower drizzle',
       'light shower snow', 'sleet'])

# Use Streamlit's date selector for the date.
selected_date = st.date_input("Select Date", value=datetime(2012, 10, 2).date())

# Use a selectbox for hour (with minutes fixed at 00)
selected_hour = st.selectbox("Select Hour (0-23)", options=list(range(24)), index=11)

# Combine date and time (with minutes fixed at 00)
selected_time = time(hour=selected_hour, minute=0)
date_time = datetime.combine(selected_date, selected_time).strftime("%Y-%m-%d %H:%M:%S")


# When the user clicks the "Predict" button, send a request to the FastAPI endpoint.
if st.button("Predict"):
    # Create a payload dictionary matching the expected input schema of your FastAPI app.
    payload = {
        "holiday": holiday,
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "weather_main": weather_main,
        "weather_description": weather_desc,
        "date_time": date_time
    }
    
    # URL for your FastAPI prediction endpoint.
    url = "http://127.0.0.1:8000/predict"
    
    # Send the POST request to the API.
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            # If successful, display the prediction.
            st.success(f"Prediction: {response.json()['predictions']}")
        else:
            # If there is an error, display the error message.
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")



