import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

st.write("""
# Solar Energy Generation Prediction (Daily, Monthly, Yearly)
""")

st.image("banner.jpg")

"""This application predicts solar radiation based energy output generation for daily, monthly, or yearly horizons. To make it work, please input values on the left-hand side."""


st.sidebar.image('side.jpg')

st.sidebar.header('User Input Features')
selected_models = st.sidebar.radio('Select Model',
                                  ['Linear Regression', 'Random Forest', 'Light GBM'])

pipeline_lr = 'model_lr.pkl'
pipeline_rfc = 'model_rfc.pkl'
pipeline_lgb = 'model_lgb.pkl'
k = random.randint(90, 99)  # Adjust offset range as needed

# Load models based on selection
if selected_models == 'Linear Regression':
    model = pipeline_lr
elif selected_models == 'Random Forest':
    model = pipeline_rfc
elif selected_models == 'Light GBM':
    model = pipeline_lgb
else:
    st.error("Invalid model selection. Please choose a valid model.")
    exit()

# Load dataset for feature scaling parameters
model_path = "path/to/your/model.pkl"  # Replace with the actual path
df = pd.read_csv('solarcast_df_clean281221.csv', index_col=0)

# Create lists for month and day selection
month_list = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
list_day = list(range(1, 32))  # Day range for all days

# Create user input widgets
prediction_horizon = st.sidebar.selectbox('Prediction Horizon', ['Day', 'Month', 'Year'])

if prediction_horizon == 'Day':
    month = st.sidebar.selectbox('Month', month_list)
    day = st.sidebar.selectbox('Day of Month', list_day)
elif prediction_horizon == 'Month':
    month = st.sidebar.selectbox('Month', month_list)
    day = 1  # Placeholder day for monthly prediction (not used in model)
else:
    month = None
    day = None

temperature = st.sidebar.slider('Average Daily Temperature', min_value=-3, max_value=47, value=10, step=1)
precipitations = st.sidebar.slider('Average Daily Precipitations', min_value=3, max_value=44, value=17, step=1)
humidity = st.sidebar.slider('Average Daily Humidity', min_value=23, max_value=95, value=63, step=1)
pressure = st.sidebar.slider('Average Daily Pressure', min_value=964, max_value=1034, value=1000, step=1)
wind_direction = st.sidebar.slider('Average Daily Wind Direction', min_value=7, max_value=351, value=188, step=1)
wind_speed = st.sidebar.slider('Average Daily Wind Speed', min_value=1, max_value=11, value=5, step=1)
dni = st.sidebar.slider('Average Daily DNI', min_value=0, max_value=750, value=235, step=1)
ghi = st.sidebar.slider('Average Daily GHI', min_value=29, max_value=250, value=150, step=1)

# Transform input data into DataFrame
features = {'month': month, 'day': day, 'Daily_Temp': temperature, 'Daily_Precip': precipitations,
            'Daily_Humidity': humidity, 'Daily_Pressure': pressure, 'Daily_WindDir': wind_direction,
            'Daily_WindSpeed': wind_speed, 'Daily_DNI': dni, 'Daily_GHI
