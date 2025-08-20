# resume_screening_app.py
import streamlit as st
import pandas as pd
import random

# Title
st.title("📄 Resume Screening - HR AI Assistant")

# Candidate input fields
name = st.text_input("Full Name")
email = st.text_input("Email")
experience = st.slider("Years of Experience", 0, 30, 2)
education = st.selectbox("Education", ["High School", "Diploma", "Bachelors", "Masters", "PhD"])
skills = st.multiselect("Skills", ["Python", "Java", "SQL", "ML/AI", "Cloud", "Communication", "Leadership"])
role = st.selectbox("Applying for Role", ["Software Engineer", "Data Scientist", "Project Manager", "HR", "Analyst"])

# Simple AI Scoring Logic (you can replace with ML model)
def calculate_fit(experience, education, skills, role):
    score = 0
    if experience >= 3:
        score += 20
    if education in ["Bachelors", "Masters", "PhD"]:
        score += 30
    if "Python" in skills and role in ["Software Engineer", "Data Scientist"]:
        score += 20
    if "Leadership" in skills and role in ["Project Manager", "HR"]:
        score += 20
    score += len(skills) * 5
    return min(score, 100)

# Predict button
if st.button("Evaluate Resume"):
    score = calculate_fit(experience, education, skills, role)
    st.subheader(f"🎯 Job Fit Score for {name}: {score}%")

    if score >= 70:
        st.success("✅ Strong Candidate - Recommend for Interview")
    elif score >= 40:
        st.warning("⚠️ Average Fit - Needs Further Screening")
    else:
        st.error("❌ Weak Fit - Not Recommended")
