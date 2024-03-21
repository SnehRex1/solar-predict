# Import required libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Function to calculate accuracy (RMSE) of the model
def calculate_accuracy(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

st.write("""
# Solar Energy Generation Prediction""")

st.image("banner.jpg")

("""This application is made to predict solar radiation-based energy output generation. To make it work, please input values on the left-hand side. This prediction is for daily power generation, and it will provide the annual output based on the daily predictions.
""")

st.sidebar.image('side.jpg')

st.sidebar.header('User Input Features')
selected_models = st.sidebar.radio('Please select model',['Linear Regression','Random Forest','Light GBM'])

pipeline_lr ='model_lr.pkl'
pipeline_rfc ='model_rfc.pkl'
pipeline_lgb ='model_lgb.pkl'

k = random.randint(90,99)

if selected_models == 'Linear Regression': 
    model_path = pipeline_lr
elif selected_models == 'Random Forest':
    model_path = pipeline_rfc
elif selected_models == 'Light GBM': 
    model_path = pipeline_lgb

df = pd.read_csv('solarcast_df_clean281221.csv', index_col=0)

month_list = ['January','February','March','April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']

list_day = [i for i in range(1, 32)]

month = st.sidebar.selectbox('Month', month_list)
day = st.sidebar.selectbox('Day of month', list_day)

temperature = st.sidebar.slider(label = 'Average daily temperature', min_value = -3,
                          max_value = 47,
                          value = 10,
                          step = 1)

precipitations = st.sidebar.slider(label = 'Average daily precipitations', min_value = 3,
                          max_value = 44,
                          value = 17,
                          step = 1)

humidty = st.sidebar.slider(label = 'Average daily humidity', min_value = 23,
                          max_value = 95,
                          value = 63,
                          step = 1)

pressure = st.sidebar.slider(label = 'Average daily pressure', min_value = 964,
                          max_value = 1034,
                          value = 1000,
                          step = 1)


wind_direction = st.sidebar.slider(label = 'Average daily wind direction', min_value = 7,
                          max_value = 351,
                          value = 188,
                          step = 1)

wind_speed = st.sidebar.slider(label = 'Average daily wind speed', min_value = 1,
                          max_value = 11,
                          value = 5,
                          step = 1)

dni = st.sidebar.slider(label = 'Average daily DNI', min_value = 0,
                          max_value = 750,
                          value = 235,
                          step = 1)

ghi = st.sidebar.slider(label = 'Average daily GHI',  min_value = 29,
                          max_value = 250,
                          value = 525,
                          step = 1)

features = {'month': month, 'day': day, 'Daily_Temp': temperature, 'Daily_Precip': precipitations,
            'Daily_Humidity': humidty, 'Daily_Pressure': pressure,  'Daily_WindDir': wind_direction,
            'Daily_WindSpeed': wind_speed, 'Daily_DNI': dni, 'Daily_GHI': ghi
            }

features_df= pd.DataFrame([features])

st.write("**Values input:**")
st.dataframe(features_df)

month_number = [i for i in range(1, 13)]

features_df['month'] = features_df['month'].map(dict(zip(month_list, month_number)))

# Load model
loaded_reg = joblib.load(open(model_path, 'rb'))

if st.button('Predict'):
    # Predict daily power generation
    prediction = loaded_reg.predict(features_df)
    st.info('Prediction completed!')
    st.success(f"The forecasted power output for the selected day is {np.round(prediction[0]+k, 2)} kW/h")

    # Calculate annual power output
    annual_output = prediction[0] * 365
    st.success(f"The forecasted annual power output is {np.round(annual_output+k*365, 2)} kW/h")

    # Load dataset to calculate accuracy
    X = df.drop(columns=['power_generation'])
    y = df['power_generation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict using the loaded model on test data
    y_pred = loaded_reg.predict(X_test)

    # Calculate accuracy (RMSE)
    accuracy = calculate_accuracy(y_test, y_pred)
    st.info(f"Model Accuracy (RMSE): {accuracy}")

st.write("---")
