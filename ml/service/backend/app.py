import streamlit as st
import requests
import base64
from PIL import Image
import io

# Configuration
# ml/service/backend/app.py
BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(layout="wide")
st.title("Fundus Multi-Model Segmentation")

uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    col_img, _ = st.columns([1, 1])  # две колонки по 50%
    with col_img:
        st.image(uploaded_file, caption="Original Image", width=500)

    if st.button("Run Segmentation"):
        with st.spinner("Processing... Please wait."):
            try:
                # Send request to backend
                response = requests.post(
                    BACKEND_URL,
                    files={"file": (uploaded_file.name,
                                    uploaded_file, uploaded_file.type)}
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data["results"]

                    # Decode Cascade Mask (Model 1 + 2)
                    cascade_bytes = base64.b64decode(results["cascade_mask"])
                    cascade_img = Image.open(io.BytesIO(cascade_bytes))

                    # Decode Model 3 Mask
                    model3_bytes = base64.b64decode(results["model_3_mask"])
                    model3_img = Image.open(io.BytesIO(model3_bytes))

                    # Display Results in Columns (по 50% каждая)
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.header("Cascade (Model 1 & 2)")
                        st.image(
                            cascade_img, caption="Binary Segmentation", width=500)

                    with col2:
                        st.header("Biomarkers (Model 3)")
                        st.image(
                            model3_img, caption="Multi-class Segmentation", width=500)

                else:
                    st.error(f"Error from server: {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
