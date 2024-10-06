import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the model from the pickle file
with open('FUEL_MODEL.pickle', 'rb') as file:
    model = pickle.load(file)

with open('SCALER_MODEL.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title("Linear Regression Prediction For MPG")

# Create input fields for your model's features
feature_1 = st.number_input("Horsepower")
# Button to make prediction
if st.button("Predict"):
    new_data = np.array([[feature_1]])  # Reshape for the scaler
    new_data_scaled = scaler.transform(new_data)  # Scale the input data
    prediction = model.predict(new_data_scaled)

    # Display the prediction
    st.write(f"Predicted MPG: {prediction[0]:.2f}")