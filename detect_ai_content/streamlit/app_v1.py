import streamlit as st
import matplotlib.pyplot as plt
import requests
import base64
import pathlib
import os

# Page Configuration
st.set_page_config(page_title="TrueNet - AI Detection", layout="wide")

# Define session_state variables
if 'text_input' not in st.session_state:
    st.session_state.text_input = "This is example of text for"

def display_results(analysis: dict):
    prediction_class = analysis["prediction"]
    prediction_proba = analysis["predict_proba"]
    prediction_proba_cleaned = float(prediction_proba.replace('%', ''))

    proba_what = "Human" if prediction_class == 0 else "AI"

    proba_human = prediction_proba_cleaned if prediction_class == 0 else 100 - prediction_proba_cleaned
    proba_ai = 100 - proba_human

    # Dynamic styling
    ring_chart_gradient_css = f"""background: conic-gradient(
            #65c6ba 0% {proba_human}%,
            #0e8c7d {proba_human}% 100%
        );
    """

    # Ring chart with labels
    st.markdown(
        f"""
        <div class="chart-container">
            <div class="ring-chart" style="{ring_chart_gradient_css}">
                <div class="chart-label">{prediction_proba_cleaned}%<br>
                    <span class="chart-label-small">{proba_what}</span>
                </div>
            </div>
            <div class="labels-container">
                <div class="label label-human">{proba_human}% Human</div>
                <div class="label label-ai">{proba_ai}% AI</div>
            </div>
        </div>
        <div class="chart-description">
            The probability this text has been entirely written by Human, Mixed, or AI.
        </div>
        """,
        unsafe_allow_html=True
    )

# Function: Analyze Text
def analyze_text(text: str) -> dict:
    headers = {'accept': 'application/json'}
    params = {"text": text}
    response = requests.get(
        'https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/text_single_predict',
        headers=headers, params=params
    )
    st.success("Prediction done ✅")
    return response.json()

def example_buttons():
    st.write("Try an example:")

    selected_example = st.radio(
        "",
        options=["Llama2", "Claude", "ChatGPT", "Human"],
        index=0,
        horizontal=True
    )

    if selected_example == "Llama2":
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=llama2_chat')
        st.session_state.text_input = response.json()['text']

    elif selected_example == "Claude":
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=darragh_claude_v6')
        st.session_state.text_input = response.json()['text']

    elif selected_example == "ChatGPT":
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=chat_gpt_moth')
        st.session_state.text_input = response.json()['text']

    elif selected_example == "Human":
        response = requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source=persuade_corpus')
        st.session_state.text_input = response.json()['text']

# Centralized CSS
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
        }

        /* Header Section */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #f5f5f5;
            padding: 10px 20px;
            border-bottom: 1px solid #ddd;
        }
        .logo img {
            height: 60px;
            width: auto;
        }
        .nav-links {
            display: flex;
            gap: 15px;
        }
        .nav-links a {
            text-decoration: none;
            color: #65c6ba;
            font-weight: bold;
        }
        .btn-dark {
            background-color: #65c6ba;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
        }

        /* Ring Chart */
        .chart-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .ring-chart {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            position: relative;
        }
        .ring-chart::before {
            content: "";
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background-color: white;
            position: absolute;
            top: 25px;
            left: 25px;
        }
        .chart-label {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 14px;
            font-weight: bold;
        }
        .chart-description {
            font-size: 12px;
            text-align: center;
        }

        /* Buttons */
        .button-container button {
            background-color: #65c6ba;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .button-container button:hover {
            background-color: #0e8c7d;
        }

        /* Footer */
        footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f5f5f5;
            padding: 10px 0;
            border-top: 1px solid #ddd;
            text-align: center;
        }
        footer a {
            color: #65c6ba;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

parent_path = pathlib.Path(__file__).parent.resolve()
save_path = os.path.join(parent_path, "data")

logo_base64_logo = get_base64_image(f"{save_path}/logo3_transparent.png")

st.markdown(
    f"""
    <div class="header">
        <div class="logo">
            <img src="data:image/png;base64,{logo_base64_logo}" alt="Your Logo">
        </div>
        <div class="nav-links">
            <a href="page_approach">Approach</a>
            <a href="page_models">Models</a>
            <a href="page_resources">Resources</a>
            <a href="page_news">News</a>
            <a href="page_team">Team</a>
        </div>
        <div class="button-container">
            <button class="btn-dark">Dashboard</button>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Divider
st.markdown("---")

# Left and Right Columns
col1, col2 = st.columns([1.5, 2])

with col1:
    st.markdown('<h1>Beyond the Surface</h1>', unsafe_allow_html=True)
    st.markdown('<h2>Identify the <span style="color: #65c6ba;">Creator</span></h2>', unsafe_allow_html=True)
    st.write(
        "Since inventing AI detection, TrueNet incorporates the latest research in detecting ChatGPT, GPT4, Google-Gemini, "
        "LLaMa, and new AI models, and investigating their sources."
    )

with col2:
    st.subheader("AI or Human?")
    example_buttons()
    st.text_area("Your text here", key="text_input", height=200)
    st.file_uploader("Upload File", type=["txt", "docx", "pdf"], key="uploader")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        if st.button("Scan for AI"):
            if st.session_state.text_input.strip():
                with st.spinner("Analyzing..."):
                    result = analyze_text(st.session_state.text_input)
                    display_results(result)
            else:
                st.warning("Please enter some text to analyze.")
    with col_right:
        if st.button("Clear"):
            st.session_state.text_input = ""
            st.session_state.selected_example = None
            st.rerun()

# Footer
st.markdown(
    """
    <footer>
        <a href="#">Contact</a> | <a href="#">Support</a> | <a href="#">Imprint</a>
        <p>© 2024 TrueNet</p>
    </footer>
    """,
    unsafe_allow_html=True
    )
