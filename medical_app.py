import streamlit as st

st.set_page_config(page_title="Medical Prediction App", page_icon="🏥")

# -----------------------------
# App Title
# -----------------------------
st.title("🏥 Medical Risk Prediction App")
st.write("Enter your health details below to check risk level and get suggestions.")

# -----------------------------
# Input Fields
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

bp = st.selectbox("Blood Pressure Level", ["Normal", "Slightly High", "High"])
sugar = st.selectbox("Blood Sugar Level", ["Normal", "Borderline", "High"])
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Borderline", "High"])
bmi = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
smoking = st.selectbox("Smoking Habit", ["No", "Occasional", "Regular"])
alcohol = st.selectbox("Alcohol Consumption", ["No", "Occasional", "Frequent"])
activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
family_history = st.selectbox("Family History of Disease", ["No", "Yes"])

# -----------------------------
# Risk Scoring Logic
# -----------------------------
risk_score = 0

# Age
if age > 50: risk_score += 2
elif age > 35: risk_score += 1

# BP
if bp == "Slightly High": risk_score += 1
elif bp == "High": risk_score += 2

# Sugar
if sugar == "Borderline": risk_score += 1
elif sugar == "High": risk_score += 2

# Cholesterol
if cholesterol == "Borderline": risk_score += 1
elif cholesterol == "High": risk_score += 2

# BMI
if bmi == "Overweight": risk_score += 1
elif bmi == "Obese": risk_score += 2

# Lifestyle
if smoking == "Regular": risk_score += 2
elif smoking == "Occasional": risk_score += 1

if alcohol == "Frequent": risk_score += 2
elif alcohol == "Occasional": risk_score += 1

if activity == "Low": risk_score += 2
elif activity == "Moderate": risk_score += 1

# Family history
if family_history == "Yes": risk_score += 2

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Risk"):
    st.subheader("Prediction Result")

    if risk_score <= 3:
        st.success("✅ Low Risk")
        st.write("👉 Suggestion: Maintain a balanced diet, stay active, and continue healthy lifestyle.")
    elif 4 <= risk_score <= 7:
        st.warning("⚠️ Medium Risk")
        st.write("👉 Suggestion: Regular exercise, reduce sugar/salt intake, and periodic health checkups.")
    else:
        st.error("🚨 High Risk")
        st.write("👉 Suggestion: Please consult a doctor for a detailed checkup. Adopt immediate lifestyle changes.")

    st.info(f"Your calculated risk score: **{risk_score}**")

