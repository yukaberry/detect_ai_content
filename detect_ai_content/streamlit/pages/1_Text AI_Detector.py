import streamlit as st
from typing import Optional
import requests
from params import *
import base64
import os
import pathlib

def get_base64_image(file_path):
    """Convert an image to a base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def local_css():
    # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_path = os.path.join(save_path, "logo3_transparent.png")

    st.markdown("""
    <style>
    /* Main Container */

    /* Header */
    .header {
        background-color: #f5f5f5;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .header img {
        height: 60px;
        width: auto;
        margin-right: 20px;
    }
    .header-title {
        flex: 1;
        text-align: center;
    }
    .header-title h2 {
        font-size: 2rem;
        font-weight: bold;
        color: #333333;
        margin: 0;
    }
    .header-title h3 {
        font-size: 1.5rem;
        color: #65c6ba;
        margin: 0;
    }

    /* Example Buttons Container */
        .example-buttons-container {
            display: flex;
            justify-content: space-around;
            margin-top: 1rem;
            gap: 0.5rem;
        }
        .example-buttons-container button {
            background-color: #65c6ba;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .example-buttons-container button:hover {
            background-color: #0e8c7d;
            cursor: pointer;
        }

    /* Container for Text Area and Buttons */
    .textarea-and-buttons-container {
        display: flex;
        flex-direction: column; /* Stack elements vertically */
        align-items: center;
        justify-content: center;
        margin-top: 1rem;
        gap: 1rem; /* Space between text area and buttons */
    }

    /* Text Area */
    .stTextArea-container {
        width: 100%; /* Make the text area take full width */
        max-width: 600px; /* Optional: Limit the max width */
        display: flex;
        justify-content: center;
        margin-bottom: 1rem; /* Space between text area and buttons */
    }

    /* Analyze Test Button (Bottom Left) */
    .analyze-button-container {
       align-self: flex-start; /* Align to the left of the container */
    }
    .analyze-button-container > button {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        background-color: #65c6ba;important!
        color: white;
        border: none;
    }
    .analyze-button-container > button:hover {
        background-color: #0e8c7d;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9rem;
        color: #777777;
    }

    /* Hide Streamlit's default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}


    </style>
    """, unsafe_allow_html=True)

    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(f"""
            <div class="header">
                <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo">
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Logo file not found. Please check the path.")

    st.markdown("""
    <h2>More than an AI Detector</h2>
    <h3>Preserve what\'s <span style="color: #65c6ba;">Human.</span></h3>

    """, unsafe_allow_html=True)

def example_buttons():
    Llama2_text = "Llama2"
    Claude_text = "Claude"
    ChatGPT_text = "ChatGPT"
    Human_text = "Human"

    option_map = {
        0: Llama2_text,
        1: Claude_text,
        2: ChatGPT_text,
        3: Human_text,
    }
    selection = st.pills(
        "Try an example:",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
    )

    if selection is not None:
        pills_selection = option_map[selection]
        if pills_selection == Llama2_text:
            source = "llama2_chat"
        elif pills_selection == Claude_text:
            source = "darragh_claude_v6"
        elif pills_selection == ChatGPT_text:
            source = "chat_gpt_moth"
        elif pills_selection == Human_text:
            source = "persuade_corpus"
        response = requests.get(f'{BASEURL}/random_text?source={source}')
        st.session_state.text_input = response.json()['text']

def text_input_section():
    return st.text_area(
        "Paste your text",
        height=200,
        key="text_input",
        help="Enter the text you want to analyze"
    )

def analyze_text(text: str) -> dict:
    """
    Placeholder for actual AI detection logic.
    """

    headers = {
        'accept': 'application/json',
    }
    params = {
        "text":text
    }
    response = requests.get(f'{BASEURL}/text_single_predict', headers=headers, params=params)
    st.success("Prediction done ✅")
    return response.json()

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
        st.metric("Predict proba", analysis["predict_proba"])

    st.markdown("#### Detailed Metrics")
    metrics = analysis["details"]
    cols = st.columns(len(metrics))
    for col, (metric, value) in zip(cols, metrics.items()):
        col.metric(
            metric.replace("_", " ").title(),
            value
        )
def create_content():
    # Store the selected example in session state if it's not already there
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

    # Get newly selected example
    new_selection = example_buttons()

    # Update session state if a new selection was made
    if new_selection is not None:
        st.session_state.selected_example = new_selection
        st.session_state.text_input = f"This is example text for {new_selection}"

    # Container for text area and buttons
    st.markdown('<div class="textarea-and-buttons-container">', unsafe_allow_html=True)

    # Text input area
    st.markdown('<div class="stTextArea-container">', unsafe_allow_html=True)
    text = text_input_section()
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyze Test button (Bottom Left)
    col1, col2 = st.columns([2, 1])
    analysis = None
    with col1:
        if st.button("Scan for AI", key="analyze", type="primary"):
            if text:
                with st.spinner("Wait for it..."):
                    analysis = analyze_text(text)
            else:
                st.warning("Please enter some text to analyze.")

    # Clear button (Bottom Right)
    with col2:
        if st.button("Clear", key="clear"):
            st.session_state.clear()
            st.session_state.text_input = ''
            st.rerun()

    if analysis is not None:
        display_results(analysis)

    # Footer
    st.markdown("""
        <div class="footer">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)


with st.container():
    local_css()
    create_content()

import requests
requests.get(f'{BASEURL}/ping')
