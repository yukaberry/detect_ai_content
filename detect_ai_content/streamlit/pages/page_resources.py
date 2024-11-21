# pages/page_approach.py
import streamlit as st
import base64
import os
import pathlib

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="TrueNet – Resources", layout="wide")

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
    ## Resources

    ### Datasets:
    - **Training Data**: The project utilizes a diverse set of datasets to train models capable of detecting AI-generated content. These datasets include both human-written and AI-generated texts to ensure comprehensive learning.
    - **Testing Data**: Separate datasets are employed to evaluate the model’s performance, ensuring that the detection system generalizes well to unseen data.

    ### Tools and Libraries:
    - **Python**: The primary programming language used for developing the detection algorithms.
    - **Streamlit**: A framework for building interactive web applications, used here to create the user interface for the AI detection tool.

    ### Machine and Deep Learning Libraries:
    - **scikit-learn**: For implementing various machine learning algorithms and evaluation metrics.
    - **TensorFlow/PyTorch**: For building and training deep learning models.
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

# This allows the resources page to be run as a standalone app for testing
if __name__ == "__main__":
    st.sidebar.title('Navigation')
    app()
