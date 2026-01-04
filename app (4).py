
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("wine_quality_model.pkl")

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("This app predicts wine quality using a regression model.")

st.subheader("Enter Wine Chemical Properties")

# Input fields (same order as training data)
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.7)
citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=1.9)
chlorides = st.number_input("Chlorides", min_value=0.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=34.0)
density = st.number_input("Density", min_value=0.9900, max_value=1.0050, value=0.9978)
pH = st.number_input("pH", min_value=2.0, max_value=4.0, value=3.51)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.56)
alcohol = st.number_input("Alcohol", min_value=0.0, value=9.4)

# Predict button
if st.button("Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                             residual_sugar, chlorides, free_sulfur_dioxide,
                             total_sulfur_dioxide, density, pH,
                             sulphates, alcohol]])

    prediction = model.predict(input_data)

    st.success(f"🍷 Predicted Wine Quality: {prediction[0]:.2f}")
