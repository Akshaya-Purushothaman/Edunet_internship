import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ==============================
# Load trained model
# ==============================
st.set_page_config(page_title="Urban Heat Island Detection", layout="centered")
st.title("🌆 Urban Heat Island (UHI) Detection using CNN")
st.write("Upload temperature and normal map images to predict UHI intensity.")

MODEL_PATH = "uhi_model.h5"

@st.cache_resource
def load_uhi_model():
    model = load_model(MODEL_PATH)
    return model

model = load_uhi_model()
IMG_SIZE = (128, 128)

# ==============================
# Upload section
# ==============================
temp_file = st.file_uploader("🌡️ Upload Temperature Map", type=["jpg", "png", "jpeg"])
normal_file = st.file_uploader("🏙️ Upload Normal (RGB) Map", type=["jpg", "png", "jpeg"])

# ==============================
# Predict function
# ==============================
def predict_uhi(temp_img, normal_img):
    temp_img = np.array(temp_img.resize(IMG_SIZE)) / 255.0
    normal_img = np.array(normal_img.resize(IMG_SIZE)) / 255.0

    # Add batch dimension
    temp_img = np.expand_dims(temp_img, axis=0)
    normal_img = np.expand_dims(normal_img, axis=0)

    pred = model.predict([temp_img, normal_img])
    classes = ['Low UHI 🌿', 'Medium UHI 🌤', 'High UHI 🔥']
    return classes[np.argmax(pred)], np.max(pred)

# ==============================
# Display and Predict
# ==============================
if temp_file and normal_file:
    temp_img = Image.open(temp_file).convert("RGB")
    normal_img = Image.open(normal_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(temp_img, caption="Temperature Map", use_container_width=True)
    with col2:
        st.image(normal_img, caption="Normal RGB Map", use_container_width=True)

    if st.button("🔍 Predict UHI Level"):
        label, confidence = predict_uhi(temp_img, normal_img)

        st.markdown("---")
        st.subheader("🧠 Model Prediction")
        st.write(f"**Predicted UHI Level:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        # Color feedback
        if "Low" in label:
            st.success("✅ Area is environmentally cool and sustainable.")
        elif "Medium" in label:
            st.warning("⚠️ Moderate heat detected — monitor vegetation cover.")
        else:
            st.error("🔥 High Urban Heat Island effect detected! Suggest increasing green areas or reflective surfaces.")

else:
    st.info("Please upload both images to begin prediction.")
