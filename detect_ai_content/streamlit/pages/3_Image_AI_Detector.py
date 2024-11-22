import streamlit as st
from PIL import Image
import numpy as np
import os
import pathlib
import io
import requests
import time
from params import *
import base64

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def display_results(analysis: dict):
    st.markdown("### Analysis Results")
    print(analysis)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", analysis["prediction"])
    with col2:
        if analysis["prediction"] == 0:
            st.metric("Prediction", "Human")
        if analysis["prediction"] == 1:
            st.metric("Prediction", "AI")

    with col3:
        probability = analysis["probability"]
        probability = 100 * probability
        st.metric("Predict proba", f"{int(probability)}%")

    st.markdown("#### Detailed Metrics")
    if "details" in analysis:
        metrics = analysis["details"]
        cols = st.columns(len(metrics))
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(
                metric.replace("_", " ").title(),
                value
            )

def app():
        # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_path = os.path.join(save_path, "logo3_transparent.png")

    # Check if the logo file exists
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(f"""
            <div style="display: flex; align-items: center; background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
                <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto; margin-right:20px;">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Logo file not found. Please check the path.")

    # Add a consistent header
    st.markdown("""
        <h2 style='text-align: left; color: black; font-size: 30px'>Is your image Real or AI Generated?</h2>
    """, unsafe_allow_html=True)


        # Custom CSS for button color
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #65c6ba; /* Base color from logo */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }
        div.stButton > button:first-child:hover {
            background-color: #4da79c; /* Slightly darker shade for hover effect */
        }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image to find out now!", type=["jpg"], accept_multiple_files=False)


    # Layout for buttons
    col1, col2, col3 = st.columns([2, 5, 1])  # Adjust column ratios
    with col1:
        # Scan for AI Button (Left)
        if st.button("Scan for AI", type="primary"):
            if uploaded_file is not None:
                st.subheader("Prediction")
                with st.spinner("Analyzing..."):
                    headers = {'accept': 'application/json'}
                    files = {'file': (uploaded_file.name, uploaded_file.read(), 'image/jpg')}
                    response = requests.post(f"{BASEURL}/image_predict_most_updated_cnn", headers=headers, files=files)
                    if response.status_code == 200:
                        st.success("Prediction completed ✅")
                        display_results(response.json())
                    else:
                        st.error("Failed to get prediction.")
            else:
                st.error("Please upload a file first.")
    with col2:
        # Empty Column (Middle) for Spacing
        st.markdown("")  # Empty space
    with col3:
        # Clear Button (Right)
        if st.button("Clear"):
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

app()
# Ping server to preload things if needed
import requests
requests.get(f'{BASEURL}/ping')
