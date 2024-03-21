# Import required libraries and packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Load dataset
df = pd.read_csv('solarcast_df_clean281221.csv', index_col=0)

# Function to calculate accuracy metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Function to train and evaluate models
def train_evaluate_model(X_train, X_test, y_train, y_test, model_type):
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'Light GBM':
        model = lgb.LGBMRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse, r2 = calculate_metrics(y_test, y_pred)
    return model, mse, r2

# Create sidebar and title
st.sidebar.title('User Input Features')
st.title('Solar Energy Generation Prediction')

# Load models
models = {
    'Linear Regression': 'model_lr.pkl',
    'Random Forest': 'model_rfr.pkl',
    'Light GBM': 'model_lgb.pkl'
}

# Get user inputs
selected_model = st.sidebar.selectbox('Please select model', list(models.keys()))
model_path = models[selected_model]
selected_date = st.sidebar.date_input('Select a date')

# Filter data for the selected date
filtered_df = df[df['Date'] == selected_date]

# Display selected date
st.write(f"Selected Date: {selected_date}")

# Display prediction for the selected date
if st.sidebar.button('Predict'):
    # Load the selected model
    loaded_model = joblib.load(open(model_path, 'rb'))
    
    # Extract features
    X = filtered_df.drop(columns=['Solar_Generation'])
    y = filtered_df['Solar_Generation']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    model, mse, r2 = train_evaluate_model(X_train, X_test, y_train, y_test, selected_model)
    
    # Predict solar generation
    prediction = model.predict(X_test)
    
    # Display prediction
    st.write(f"Predicted Solar Generation for {selected_date}: {prediction} kW/h")
    st.write(f"Model Accuracy (MSE): {mse}")
    st.write(f"Model Accuracy (R^2 Score): {r2}")
