# pages/page_approach.py

import streamlit as st
import base64
import os
import pathlib

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

    # Main content of the page
    st.markdown("""
    ## Our Approach

    ### Data Collection and Preprocessing:
    - **Data Gathering**: Collecting a balanced dataset comprising both AI-generated and human-written content.
    - **Preprocessing**: Cleaning the data by removing noise, normalizing text, and tokenizing sentences to prepare it for model training.

    ### Model Development:
    - **Feature Extraction**: Utilizing techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features that machine learning models can process.
    - **Model Selection**: Experimenting with various algorithms, including logistic regression, support vector machines, and neural networks, to identify the most effective model for detecting AI-generated content.

    ### Training and Evaluation:
    - **Training**: Feeding the preprocessed data into the selected models and adjusting parameters to optimize performance.
    - **Evaluation**: Assessing model accuracy, precision, recall, and F1-score using the testing dataset to ensure reliability.

    ### Deployment:
    - **Integration with Streamlit**: Developing an interactive web application that allows users to input text and receive real-time analysis on whether the content is AI-generated.
    - **User Interface Design**: Creating a user-friendly interface that displays results clearly and provides insights into the detection process.
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

app()
