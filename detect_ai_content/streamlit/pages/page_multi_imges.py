import streamlit as st
import base64
import os
import pathlib
import requests
import time
from params import *

# # Set page configuration
# st.set_page_config(page_title="TrueNet – Image AI Detection", layout="wide")

# # Convert the image to Base64
# def get_base64_image(file_path):
#     with open(file_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# def app():
#     # Prepare base64 string for the logo
#     parent_path = pathlib.Path(__file__).parent.parent.resolve()
#     save_path = os.path.join(parent_path, "data")
#     logo_base64 = get_base64_image(f"{save_path}/logo3_transparent.png")

#     # Display custom header with logo
#     st.markdown(f"""
#         <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
#             <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto;">
#         </div>
#     """, unsafe_allow_html=True)

#     # Title
#     st.markdown("""
#         ## **Is your image Human or AI Generated?**
#     """)

#     # Instruction
#     st.write("""
#     Upload an image below to find out if it's created by AI or a human:
#     """)

#     # File uploader
#     image = st.file_uploader("", type=["JPG", "JPEG", "PNG"])

#     # Run Prediction Button

#     if st.button("Run Prediction") and image is not None:
#         # API URL (Replace with actual deployed API)
#         # TODO
#         api_url = "http://0.0.0.0:8000/image_multi_predict"
#         files = {"file": image.getvalue()}

#         try:
#             response = requests.post(api_url, files=files)

#             if response.status_code == 200:
#                 prediction = response.json().get("prediction", "No prediction found")
#                 st.markdown(f"""
#                     <div style="background-color:#65c6ba; color:white; padding:10px; border-radius:5px; text-align:center; font-size:18px; margin-top:20px;">
#                         Prediction complete ✅ <br><b>{prediction}</b>
#                     </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.error("Error: " + response.text)
#         except Exception as e:
#             st.error("Error while connecting to the API. Please try again later.")

#     # Footer
#     st.markdown("---")
#     st.markdown("""
#         <div style="text-align:center;">
#             © 2024 TrueNet AI Detection. All rights reserved.
#         </div>
#     """, unsafe_allow_html=True)


# def analyze_img(img: str) -> dict:

#     """
#     Placeholder for actual AI detection logic.
#     """

#     headers = {
#         'accept': 'application/json',
#     }
#     params = {
#         "text":text
#     }
#     response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/text_multi_predict', headers=headers, params=params)
#     st.success("Prediction done ✅")
#     return response.json()

# def main():
#     analyze_img()





# # This allows the app to run as a standalone page for testing
# if __name__ == "__main__":
#     #st.sidebar.title('Navigation')

#     main()

# import requests
# requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/ping')


from PIL import Image
import pandas as pd
import streamlit as st

def image_input_section():
    return st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="image_input",
        help="Upload the image you want to analyze"
    )


def analyze_image(image_file) -> dict:
    """
    Sends the uploaded image to the image classification endpoint.
    """
    headers = {
        'accept': 'application/json',
    }
    files = {
        'user_input': (image_file.name, image_file, image_file.type)
    }
    response = requests.post(
         f'{BASEURL}/image_multi_predict',
        # local test
        #'http://0.0.0.0:8080/image_multi_predict',
        headers=headers,
        files=files
    )
    try:
        response.raise_for_status()  # Ensure no HTTP error occurred
        return response.json()  # Try to parse JSON response
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred: {e}")
    except ValueError:
        # Response is not JSON
        st.error(f"Server response is not in JSON format: {response.text}")
    return None

def display_results(analysis: dict):
    st.markdown("### Analysis Results")
    print(analysis)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.metric("Prediction", analysis["prediction"])
    # with col2:
    #     st.metric("Predict proba", analysis["predict_proba"])

    # st.markdown("#### Detailed Metrics")
    # details = analysis["details"]
    # models = details["models"]
    # df = pd.DataFrame(data=models)
    # df = df.T
    # df = df[['predict_proba_class', 'predicted_class']]
    # st.table(df)

    col1, col2 , col3= st.columns(3)

    with col1:
        st.metric("Prediction Class", analysis["prediction"])

    with col2:
        if analysis["prediction"] == 0:

            st.metric("Prediction", "Human")
        if analysis["prediction"] == 1:

            st.metric("Prediction", "AI")

    with col3:
        st.metric("Predict proba", analysis["predict_proba"])

    st.markdown("#### Detailed Metrics")
    details = analysis["details"]
    models = details["models"]
    df = pd.DataFrame(data=models)
    df = df.T
    df = df[['predict_proba_class', 'predicted_class']]

    # rename columns for urser
    df.columns = ['Probability', 'Predicted Class']
    df['AI or Human'] = df['Predicted Class'].apply(lambda x: "Human" if x == 0 else "AI")

     # Reset the index so that model names become a column (so we can style them)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model Name'}, inplace=True)


    def highlight_ai(row):
            # Use a light red color for highlighting if 'predicted_class' is 1
            style = [''] * len(row)  # Default style: no highlight
            if row['Predicted Class'] == 1:
                # Highlight the entire row with light red if predicted_class is AI
                style = ['background-color: #FFE6E6'] * len(row)

            # Highlight the model name (now a column) with a different color if it's predicted as AI
            if row['Predicted Class'] == 1:
                #  Soft red for model names (first column)
                style[0] = 'background-color: #FFE6E6'  #

            return style


    df = df.set_index('Model Name')
    # Apply the style function to the DataFrame
    styled_df = df.style.apply(highlight_ai, axis=1)

    st.write(styled_df, use_container_width=True)



def create_content():
    st.title("AI Detector for Images")
    st.markdown("### Detect Fake vs Real Images.")
    st.write("Upload an image to analyze whether it is likely AI-generated or real.")

    # Image upload section
    image_file = image_input_section()

    col1, col2 = st.columns([2, 1])
    analysis = None

    with col1:
        if st.button("Scan for AI", type="primary"):
            if image_file:
                with st.spinner('Analysing...'):
                    analysis = analyze_image(image_file)
            else:
                st.warning("Please upload an image to analyse.")
    with col2:
        if st.button("Clear", type="secondary"):
            st.session_state.clear()
            st.session_state.image_input = None
            st.rerun()

    if analysis is not None:
        display_results(analysis)

with st.container():
    create_content()
