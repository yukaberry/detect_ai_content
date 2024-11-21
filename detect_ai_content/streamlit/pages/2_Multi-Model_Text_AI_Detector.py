import streamlit as st
from typing import Optional
import requests
import pandas as pd
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
        body {
            font-family: 'Arial', sans-serif;
        }

        /* Main Container */
        .main {
            padding: 2rem;

        }

        /* Header */

        .header {
            background-color: #f5f5f5;
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .header img { /*logo*/
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

        /* Top Buttons */

        .stButton>button {
            border-radius: 4px;
            padding: 0.5rem 1rem;
            background-color: #65c6ba !important; /* Blue from logo */
            color: white !important;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0e8c7d !important; /* Darker blue */
            color: white !important;
        }

        /* Analyze and Clear Buttons */

        .stButton.analyze {
            background: #65c6ba;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 10px;
            color: white;
            font-size: 1rem;
            margin: 1rem 0;
        }
        .stButton.clear {
            background: #f5f5f5;
            border: 1px solid #d9d9d9;
            padding: 0.5rem 1.5rem;
            border-radius: 10px;
            color: #333333;
            font-size: 1rem;
            margin: 1rem 0;
        }
        .stButton.clear:hover {
            background: #e0e0e0;
        }
        /* Text Input */

        .stTextArea textarea {
            min-height: 150px; /* Adjust the height as needed */
            padding: 1rem; /* Add some inner spacing */
            font-size: 1rem; /* Text size */
            border: none; /* Remove the border */
            outline: none; /* Remove focus outline */
            background-color: #f5f5f5; /* Optional: Light gray background for distinction */
            border-radius: 10px; /* Optional: Rounded corners for a cleaner look */
        }

        .stTextArea textarea:focus {
            border: none; /* Ensure no border appears on focus */
            outline: none; /* Remove any browser-generated focus outline */
            box-shadow: none; /* Prevent any shadow effects */
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
    st.write("Try an example:")

    # Add a container for custom button styling
    st.markdown('<div class="example-buttons-container">', unsafe_allow_html=True)

    # Use columns within the container
    cols = st.columns(4)

    # Track which button was clicked
    selected_example = None

    # Create all buttons and check their states
    if cols[0].button("Llama2", key="example1", type="secondary"):
        response = requests.get(f'{BASEURL}/random_text?source=llama2_chat')
        print(response.json())
        selected_example = response.json()['text']

    if cols[1].button("Claude", key="example2", type="secondary"):
        response = requests.get(f'{BASEURL}/random_text?source=darragh_claude_v6')
        print(response.json())
        selected_example = response.json()['text']

    if cols[2].button("ChatGPT", key="example3", type="secondary"):
        response = requests.get(f'{BASEURL}/random_text?source=chat_gpt_moth')
        print(response.json())
        selected_example = response.json()['text']

    if cols[3].button("Human", key="example4", type="secondary"):
        response = requests.get(f'{BASEURL}/random_text?source=persuade_corpus')
        print(response.json())
        selected_example = response.json()['text']

    return selected_example


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
    response = requests.get(f'{BASEURL}/text_multi_predict', headers=headers, params=params)
    st.success("Prediction done ✅")
    return response.json()

def display_results(analysis: dict):
    st.markdown("### Analysis Results")
    print(analysis)

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

    # st.table(df)
    # st.dataframe(df, use_container_width=True)

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
    # Store the selected example in session state if it's not already there
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

    # Get newly selected example
    new_selection = example_buttons()

    # Update session state if a new selection was made
    if new_selection is not None:
        st.session_state.selected_example = new_selection
        st.session_state.text_input = f"This is example text for {new_selection}"

    # Display the text input area
    text = text_input_section()

    # Bottom buttons for Analyze and Clear
    st.markdown('<div class="bottom-buttons-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])  # Two equally spaced columns
    with col1:
        if st.button("Analyze Test", type="primary", key="analyze"):
            if text:
                with st.spinner("Wait for it..."):
                    analysis = analyze_text(text)
                    display_results(analysis)
            else:
                st.warning("Please enter some text to analyze.")
    with col2:
        if st.button("Clear", type="secondary", key="clear"):
            st.session_state.clear()
            st.session_state.text_input = ''
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

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
