import streamlit as st
from typing import Optional
import requests
import pandas as pd
from params import *
import base64
import os
import pathlib

import numpy as np
import plotly.graph_objects as go

def plot_page_results(pages_predicted_classes):

    # Définir la taille de la grille
    rows, cols = 5, 20  # 5 lignes (par exemple pour les jours) et 20 colonnes

    items = pages_predicted_classes
    missing_elements = (rows*cols) - len(items)
    if missing_elements > 0:
        for i in range(missing_elements):
            items.append(-1)

    # Générer des données aléatoires pour les nuances de couleurs
    data = np.array(pages_predicted_classes)

    data = data.reshape((rows, cols))
    print(data)

    # Créer une figure Plotly
    fig = go.Figure()

    # Ajouter des carrés à la figure
    for i in range(rows):
        for j in range(cols):
            color = "rgba(255, 230, 230, 1)" # red
            if data[i, j] == -1:
                color = "rgba(244, 250, 252, 0.5)"
            if data[i, j] == 0:
                color = "rgba(223, 253, 233, 1)" # green
            fig.add_shape(
                type="rect",
                x0=j, y0=rows - i - 1, x1=j+1, y1=rows - i,  # Ajustement pour commencer par le haut
                line=dict(color="black", width=0.5),
                fillcolor=color
            )


    # Ajuster les axes pour que cela ressemble à une grille
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, cols]
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, rows]
    )

    # Ajuster la mise en page
    fig.update_layout(
        width=800,
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="black"
    )

    # Afficher avec Streamlit
    st.plotly_chart(fig)

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

        /* Custom button styles */

        div.stButton > button {
            background-color: #65c6ba; /* Base color */
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        div.stButton > button:hover {
            background-color: #0e8c7d; /* Slightly darker shade for hover effect */
        }

        /* Align Clear button to the right */
        .clear-button-container {
            text-align: right; /* Align Clear button to the right */
        }

        /* end of button de m...styling :x */

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

def pdf_input_section():
    return st.file_uploader(
        "Upload a book",
        type=["pdf"],
        key="pdf_input",
        help="Upload a book you want to analyze"
    )


def analyze_pdf(file) -> dict:
    """
    Sends the uploaded image to the image classification endpoint.
    """
    headers = {
        'accept': 'application/json',
    }
    files = {
        'file': (file.name, file, file.type)
    }
    response = requests.post(
        'http://127.0.0.1:8000/text_book_predict',
        #  f'{BASEURL}/text_book_predict',
        headers=headers,
        files=files
    )
    try:
        response.raise_for_status()  # Ensure no HTTP error occurred
        st.session_state.analysis = response.json()
        return
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred: {e}")
    except ValueError:
        # Response is not JSON
        st.error(f"Server response is not in JSON format: {response.text}")
    return None


def display_summary_results(analysis: dict):
    st.markdown("### Analysis detailed Results")

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

def display_results(analysis: dict):
    st.markdown("### Analysis detailed Results")

    details = analysis["details"]
    pages = details["pages"]
    print(pages)
    df = pd.DataFrame(data=pages)

    # rename columns for user
    df.columns = ['Predicted Class', 'Probability', 'Model', 'Page']
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
    # Store the selected example in session state if it's not already there
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

    if 'analysis' not in st.session_state:
        st.session_state.analysis = None


    # Get newly selected example
    pdf_file = pdf_input_section()

    # Create button layout
    col1, col2 = st.columns([3, 1])  # Adjust column ratios as needed
    with col1:
        if st.button("Scan for AI", type="primary"):
            if pdf_file:
                with st.spinner('Analysing...'):
                    analysis = analyze_pdf(pdf_file)
            else:
                st.warning("Please upload a PDF to analyze.")
    with col2:
        # Wrap the Clear button inside a div for alignment
        st.markdown('<div class="clear-button-container">', unsafe_allow_html=True)
        if st.button("Clear", type="secondary"):
            st.session_state.clear()
            st.session_state.image_input = None
            st.session_state.analysis = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Display analysis results if available
    if st.session_state.analysis is not None:
        analysis = st.session_state.analysis
        details = analysis["details"]
        pages = details["pages"]
        predictions = []
        for page in pages:
            predictions.append(page['predicted_class'])

        display_summary_results(analysis)

        st.markdown("### Pages - Results")
        plot_page_results(predictions)

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
