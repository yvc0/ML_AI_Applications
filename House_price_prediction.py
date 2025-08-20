import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Step 1: Generate Synthetic Hyderabad Dataset
# -----------------------------
np.random.seed(42)
n_samples = 500

# Features
area = np.random.randint(500, 4000, n_samples)          # in sqft
bedrooms = np.random.randint(1, 6, n_samples)           # 1 to 5
bathrooms = np.random.randint(1, 5, n_samples)          # 1 to 4

# Hyderabad Localities
locations = ["Madhapur", "Kukatpally", "Kondapur", "Gachibowli", "Miyapur"]
location = np.random.choice(locations, n_samples)

# Base price per sqft by locality
price_map = {
    "Madhapur": 7500,
    "Kukatpally": 6000,
    "Kondapur": 7000,
    "Gachibowli": 8000,
    "Miyapur": 5500
}

# Target: Price
price = []
for i in range(n_samples):
    base = area[i] * price_map[location[i]]
    bonus = (bedrooms[i] * 500000) + (bathrooms[i] * 300000)
    noise = np.random.randint(100000, 500000)
    price.append(base + bonus + noise)

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Location": location,
    "Price": price
})

# -----------------------------
# Step 2: Train Model
# -----------------------------
X = df[["Area", "Bedrooms", "Bathrooms", "Location"]]
y = df["Price"]

preprocessor = ColumnTransformer(
    transformers=[('loc', OneHotEncoder(), ["Location"])],
    remainder='passthrough'
)

model = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 3: Streamlit UI
# -----------------------------
st.title("🏠 Hyderabad House Price Prediction")
st.write("Enter house details below to predict price")

# Inputs
area_input = st.number_input("Area (in sqft)", min_value=500, max_value=10000, value=1500)

bedrooms_input = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
bathrooms_input = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
location_input = st.selectbox("Select Location", locations)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[area_input, bedrooms_input, bathrooms_input, location_input]],
                               columns=["Area", "Bedrooms", "Bathrooms", "Location"])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated House Price in {location_input}: ₹ {predicted_price:,.0f}")

# -----------------------------
# Step 4: Model Performance
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**MAE:** {mae:,.0f}")
st.sidebar.write(f"**R² Score:** {r2:.2f}")
