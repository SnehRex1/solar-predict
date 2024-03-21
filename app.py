import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

st.write("""
# Solar Energy Generation Prediction (Daily & Annual)
""")

st.image("banner.jpg")

"""This application is designed to predict solar radiation-based energy output generation for both daily and annual periods. Enter your desired values on the left side.

**Note:** This is currently a hypothetical example, as daily prediction accuracy metrics may not be readily available in the provided models."""


st.sidebar.image('side.jpg')

st.sidebar.header('User Input Features')
selected_models = st.sidebar.radio('Please select model:',
                                  ['Linear Regression', 'Random Forest', 'Light GBM'])

pipeline_lr = 'model_lgb.pkl'  # Assuming all models are saved in the same file
pipeline_rfc = 'model_lgb.pkl'
pipeline_lgb = 'model_lgb.pkl'

# Load the model based on selection (can be improved for separate models)
model = joblib.load(open(pipeline_lgb, 'rb'))

# Load dataset to determine feature ranges
model_path = "path/to/your/model.pkl"  # Replace with the actual model file path
df = pd.read_csv('solarcast_df_clean281221.csv', index_col=0)

# Create lists for month and day selection
month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']

list_day = list(range(1, 32))  # Day range (1-31)

# User input widgets
month = st.sidebar.selectbox('Month', month_list)
day = st.sidebar.selectbox('Day of month', list_day)

temperature = st.sidebar.slider(label='Average daily temperature', min_value=-3,
                               max_value=47, value=10, step=1)
precipitations = st.sidebar.slider(label='Average daily precipitations', min_value=3,
                                 max_value=44, value=17, step=1)
humidty = st.sidebar.slider(label='Average daily humidity', min_value=23,
                           max_value=95, value=63, step=1)
pressure = st.sidebar.slider(label='Average daily pressure', min_value=964,
                             max_value=1034, value=1000, step=1)
wind_direction = st.sidebar.slider(label='Average daily wind direction', min_value=7,
                                   max_value=351, value=188, step=1)
wind_speed = st.sidebar.slider(label='Average daily wind speed', min_value=1,
                               max_value=11, value=5, step=1)
dni = st.sidebar.slider(label='Average daily DNI', min_value=0, max_value=750,
                        value=235, step=1)
ghi = st.sidebar.slider(label='Average daily GHI', min_value=29, max_value=250,
                        value=525, step=1)

# Function to transform input data into a DataFrame
def create_features_df(month, day, temperature, precipitations, humidty, pressure,
                       wind_direction, wind_speed, dni, ghi):
    features = {'month': month, 'day': day, 'Daily_Temp': temperature,
                 'Daily_Precip': precipitations, 'Daily_Humidity': humidty,
                 'Daily_Pressure': pressure, 'Daily_WindDir': wind_direction,
                 'Daily_WindSpeed': wind_speed, 'Daily_DNI': dni, 'Daily_GHI': ghi}
    return pd.DataFrame([features])

# Display user-selected inputs
features_df = create_features_df(month, day, temperature, precipitations, humidty,
                                 pressure, wind_direction, wind_speed, dni, ghi)
st.write("**Values Inputed:**")
st.dataframe
