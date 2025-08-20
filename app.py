import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Train model and return both model and accuracy
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv", header=None)
    df.columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    pipe.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, accuracy

# Load model
model, accuracy = train_model()

# UI layout
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺Venkat Designed Diabetes Prediction App")
st.markdown("Enter the patient's health information below:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of being diabetic

    if prediction == 1:
        st.error("🔴 The patient is likely **Diabetic**.")
    else:
        st.success("🟢 The patient is **Not Diabetic**.")

    st.markdown(f"**Prediction Confidence:** {probability * 100:.2f}%")
    st.markdown(f"**Model Accuracy:** {accuracy * 100:.2f}% (on test data)")
