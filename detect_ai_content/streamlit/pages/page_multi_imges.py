import streamlit as st
import base64
import os
import pathlib
import requests
import time

# Set page configuration
st.set_page_config(page_title="TrueNet – Image AI Detection", layout="wide")

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def app():
    # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_base64 = get_base64_image(f"{save_path}/logo3_transparent.png")

    # Display custom header with logo
    st.markdown(f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto;">
        </div>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
        ## **Is your image Human or AI Generated?**
    """)

    # Instruction
    st.write("""
    Upload an image below to find out if it's created by AI or a human:
    """)

    # File uploader
    image = st.file_uploader("", type=["JPG", "JPEG", "PNG"])

    # Run Prediction Button
    # TODO @Lina can you make it green?
    if st.button("Run Prediction") and image is not None:
        # API URL (Replace with actual deployed API)
        api_url = "https://detect-ai-content-image-api-334152645738.europe-west1.run.app/predict"
        files = {"file": image.getvalue()}

        try:
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                prediction = response.json().get("prediction", "No prediction found")
                st.markdown(f"""
                    <div style="background-color:#65c6ba; color:white; padding:10px; border-radius:5px; text-align:center; font-size:18px; margin-top:20px;">
                        Prediction complete ✅ <br><b>{prediction}</b>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Error: " + response.text)
        except Exception as e:
            st.error("Error while connecting to the API. Please try again later.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

# This allows the app to run as a standalone page for testing
if __name__ == "__main__":
    st.sidebar.title('Navigation')
    app()
