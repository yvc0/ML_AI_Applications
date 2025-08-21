import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import time

# -----------------------------
# Placeholder disease detection function
# (Replace with your ML/DL model)
# -----------------------------
def predict_disease(image):
    # Simulate model prediction
    time.sleep(1)
    diseases = {
        "Healthy": "✅ No disease detected. Maintain good irrigation & crop care.",
        "Tomato Mosaic Virus": "🦠 Isolate infected plants, use virus-free seeds, and disinfect tools.",
        "Powdery Mildew": "🌿 Use sulfur-based fungicides. Avoid overhead irrigation.",
        "Leaf Blight": "☣️ Apply copper-based fungicides. Remove infected leaves."
    }
    pred_label = np.random.choice(list(diseases.keys()))
    confidence = round(np.random.uniform(70, 99), 2)
    advice = diseases[pred_label]
    return pred_label, confidence, advice

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🌾 Crop Disease Detection", layout="wide")

st.title("🌾 Crop Disease Detection & Advisory Dashboard")
st.markdown("Upload a leaf image to detect crop disease and get treatment advice. "
            "⚠️ *This is a demo template. Replace with your own trained model for real predictions.*")

# Sidebar
st.sidebar.header("Options")
crop_type = st.sidebar.selectbox("Select Crop", ["Tomato", "Rice", "Wheat", "Maize", "Other"])
show_history = st.sidebar.checkbox("Show Prediction History")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Uploaded {crop_type} Leaf", use_container_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("Analyzing leaf..."):
            label, conf, advice = predict_disease(image)

        st.success(f"Prediction: **{label}** ({conf}%)")
        st.info(f"💡 Recommendation: {advice}")

        # Save results to session state
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(
            {"Crop": crop_type, "Disease": label, "Confidence": conf}
        )

# Show history of predictions
if show_history and "history" in st.session_state:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
    st.bar_chart(df["Disease"].value_counts())

# Footer
st.markdown("---")
st.caption("⚠️ Educational demo only. Not a medical/agriculture diagnostic tool.")
