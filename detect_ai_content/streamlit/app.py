import streamlit as st
import requests

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000 ;
        color: #FFFFFF
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Detect AI Content")
st.divider()

# Text section
st.subheader('Text')
st.file_uploader("Upload Text Data Here")
st.button("Let's Predict!")

st.divider()

# Image section
st.subheader('Image')
image = st.file_uploader("Upload your Image Here", type=["jpg", "jpeg", "png"])

# API URL will need to be replaced by the service URL that Yuka will generate after deploying to Cloud Run
if st.button("Hit me!") and image is not None:
    api_url = "https://detect-ai-content-image-api-334152645738.europe-west1.run.app/predict"
    files = {"file": image.getvalue()}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        st.success("Prediction: " + response.json().get("prediction", "No prediction found"))
    else:
        st.error("Error: " + response.text)
