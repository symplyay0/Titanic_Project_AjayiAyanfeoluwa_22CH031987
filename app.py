# app.py
import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model and scaler
# =========================
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# =========================
# App title
# =========================
st.title("Titanic Survival Prediction System")
st.write("Enter passenger details to predict survival:")

# =========================
# User Inputs
# =========================
pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=32.2)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# =========================
# Encode categorical inputs
# =========================
sex_encoded = 0 if sex == 'male' else 1
embarked_encoded = {'C':0, 'Q':1, 'S':2}[embarked]

# Create DataFrame for model input
input_data = pd.DataFrame([[pclass, sex_encoded, age, fare, embarked_encoded]],
                          columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked'])

# Scale input
input_scaled = scaler.transform(input_data)

# =========================
# Prediction Button
# =========================
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")
