# pages/page_models.py
import streamlit as st
import base64
import os
import pathlib

import streamlit as st
import base64
import os
import pathlib

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def app():

    # Prepare Base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_base64 = get_base64_image(f"{save_path}/logo3_transparent.png")

    st.markdown("""
   <style>
    .streamlit-expanderHeader {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #65c6ba !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:20px; display:flex; align-items:center;">
        <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; margin-right:20px;">
    </div>
    """, unsafe_allow_html=True)

    # Page Title
    st.markdown('<h2 style="text-align:left; font-weight:bold;">Our Models</h2>', unsafe_allow_html=True)

    # Baseline Models
    with st.expander("Baseline Models", expanded=False):
        st.markdown("""
         ##Logistic Regression
        - **Overview:** A simple yet effective statistical model used as a baseline for comparison with more advanced algorithms.
        - **Purpose:** Measures the relationship between input features and the likelihood of text being AI-generated or human-written.
        - **Advantages:** Computationally efficient, easy to implement, and interpretable.

         ##Support Vector Machines (SVM)
        - **Overview:** A robust supervised learning algorithm for text classification.
        - **Kernel Trick:** Projects data into higher-dimensional spaces for better classification.
        - **Advantages:** Works well with small-to-medium datasets and captures complex patterns.
        """, unsafe_allow_html=True)

    with st.expander("Advanced Models", expanded=False):
        st.markdown("""
        ### Neural Networks
        - **Overview:** Mimics human brain functionality to detect nuanced differences in text.
        - **Applications:** Capturing nonlinear patterns in text data for accurate detection.
        - **Architectures:**
            - **Feedforward Networks:** For straightforward classification.
            - **RNNs:** To handle sequential text data effectively.

        ### Transformer-Based Models
        - **Overview:** Leverages models like BERT for deep contextual understanding.
        - **Advantages:** Captures both preceding and succeeding context for enhanced classification.
        - **Use Cases:** State-of-the-art text classification tasks.
        """, unsafe_allow_html=True)

    # Model Performance Section
    st.markdown("""
    ### Model Performance
    - **Accuracy:** Overall correctness of predictions.
    - **Precision:** Focuses on the quality of positive predictions.
    - **Recall:** Measures the ability to capture actual AI-generated texts.
    - **F1-Score:** A harmonic mean of precision and recall.
    """)

    # Confusion Matrix Example Section
    st.markdown("### Model Benchmark:")
    confusion_matrix_path = os.path.join(save_path, "model_benchmark.png")
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption="Model Benchmark", use_column_width=False, width=700)
    else:
        st.warning("Confusion Matrix image not found. Please check the path.")

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:40px; font-size:14px; color:#777;">
        © 2024 TrueNet AI Detection. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

app()
