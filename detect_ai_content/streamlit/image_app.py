import streamlit as st
import requests
import time

st.set_page_config(
    page_title="TrueNet - AI Detection",
    layout="wide",
    initial_sidebar_state="auto")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/premium-photo/abstract-light-blue-background_142396-1043.jpg'); /* Background image */
        background-size: cover; /* Makes the background image cover the entire page */
        background-position: center; /* Centers the background image */
        color: #141414; /* Dark grey text for readability */
    }
    .stButton>button {
        background-color: #C30010;
        color: #ffffff;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader label {
        color: #141414;
        font-weight: bold;
        font-size: 16px;
    }
    hr {
        border-top: 2px solid #141414;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and divider
st.title("Is your image Real or AI Generated?")
st.divider()

st.markdown(
    """
    <h3 style="color:#141414; font-weight: bold;">Upload image to find out now!</h3>
    """, unsafe_allow_html=True
)

# Upload image and run prediction
image = st.file_uploader("",type=["JPG", "JPEG", "PNG"])

# API URL will need to be replaced by the service URL that Yuka will generate after deploying to Cloud Run
if st.button("Run Prediction") and image is not None:
    api_url = "https://detect-ai-content-image-api-334152645738.europe-west1.run.app/predict"
    files = {"file": image.getvalue()}
    response = requests.post(api_url, files=files)

    if response.status_code == 200:
        st.markdown('<p style="color:#141414; font-weight: bold; background-color: #00FF00; padding: 10px 20px; border-radius: 5px; font-size: 16px;">Prediction complete ✅</p>', unsafe_allow_html=True)
        time.sleep(2)
        prediction_text = "Model has predicted your image to be: " + response.json().get("prediction", "No prediction found")
        st.markdown(f'<p style="color:#141414; font-weight: bold; background-color: #00FF00; padding: 10px 20px; border-radius: 5px; font-size: 16px;">{prediction_text}</p>', unsafe_allow_html=True)
    else:
        st.error("Error: " + response.text)
