import streamlit as st
import base64
import os
import pathlib
import pandas as pd
import requests

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="TrueNet – AI Text Analyzer", layout="wide")

def get_base64_image(file_path):
    """Convert an image to a base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def app():
    # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_path = os.path.join(save_path, "logo3_transparent.png")

    st.markdown(
        """
        <style>
        /* General Styles */
        body {
            font-family: Arial, sans-serif;
        }
        /* Header */
        .header {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .header img {
            height: 60px;
            width: auto;
            margin-right: 10px;
        }
        .header-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            padding: 0.5rem;
        }
        .example-btn {
            margin: 0.25rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            background: white;
            cursor: pointer;
        }
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #777;
        }
        /* Hide Streamlit's default menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display custom header with logo
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(f"""
            <div class="header">
                <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo">
                <span class="header-title">TrueNet AI Text Analyzer</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Logo file not found. Please check the path.")

    # Main content of the page
    st.markdown("""
        ## More than an AI Detector
        ### Preserve what's human.

        Since inventing AI detection, we incorporate the latest research in detecting ChatGPT, GPT4, Google-Gemini, LLaMa, and new AI models, and investigating their sources.
    """)

    # Example buttons for selecting text sources
    st.write("Try an example:")
    cols = st.columns(4)

    selected_example = None
    if cols[0].button("Llama2"):
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=llama2_chat')
        selected_example = response.json().get('text', '')

    if cols[1].button("Claude"):
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=darragh_claude_v6')
        selected_example = response.json().get('text', '')

    if cols[2].button("ChatGPT"):
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=chat_gpt_moth')
        selected_example = response.json().get('text', '')

    if cols[3].button("Human"):
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=persuade_corpus')
        selected_example = response.json().get('text', '')

    # Text input and analyze/clear buttons
    text = st.text_area("Paste your text", height=200, key="text_input", help="Enter the text you want to analyze")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Analyze"):
            if text:
                with st.spinner('Analyzing...'):
                    headers = {'accept': 'application/json'}
                    params = {"text": text}
                    response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/text_single_predict', headers=headers, params=params)
                    analysis = response.json()
                    st.write("### Analysis Results")
                    st.write(analysis)
            else:
                st.warning("Please enter some text to analyze.")
    with col2:
        if st.button("Clear"):
            st.session_state.text_input = ""
            st.rerun()

    # Footer
    st.markdown("""
        <div class="footer">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
