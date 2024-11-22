# app_v0.py
import streamlit as st
import io
import requests
import base64
import pathlib
import os
from params import *

# Page Configuration
st.set_page_config(page_title="TrueNet - AI Detection", layout="wide", initial_sidebar_state="collapsed")

# CSS for Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }

    /* Header */
    .header {
        display: flex;
        align-items: center;
        background-color: #f5f5f5;
        padding: 10px 20px;
        border-bottom: 1px solid #ddd;
    }

    /* Logo */
    .logo img {
        height: 60px;
        width: auto;
        margin-right: 20px;
    }

    /* Navigation Links */
    .nav-links {
        display: flex;
        gap: 20px; /* Space between links */
        margin-left: 20px; /* Space between the logo and the links */
    }

    .nav-links a {
        text-decoration: none;
        color: #65c6ba; /* Base color */
        font-weight: bold;
        transition: color 0.3s;
    }

    .nav-links a:hover {
        color: #0e8c7d; /* Darker shade on hover */
    }


    /* End of Header */

    /* END Style for the file uploader button */

    /* START Style for CHARTS */
    .chart-container {
        display: flex;
        flex-direction: row; /* Horizontal layout for chart and labels */
        align-items: center;
        justify-content: center;
        margin-top: 20px;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        width: 100%;
    }

    /* Ring chart styling */
    .ring-chart {
        width: 150px; /* Chart size */
        height: 150px;
        border-radius: 50%;
        background: conic-gradient(
            #65c6ba 0% 60%, /* Dark green for Human */
            #eed20a 60% 70%, /* Purple for Mixed */
            #0e8c7d 70% 100% /* Gold for AI */
        );
        position: relative;
        margin-right: 20px; /* Space between chart and labels */
    }

    /* Inner circle for ring effect */
    .ring-chart::before {
        content: "";
        width: 100px; /* Inner circle size */
        height: 100px;
        border-radius: 50%;
        background-color: white;
        position: absolute;
        top: 25px;
        left: 25px;
    }

    /* Center text inside the ring */
    .chart-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px;
        font-weight: bold;
        color: #333;
        text-align: center;
    }

    /* Labels container */
    .labels-container {
        display: flex;
        flex-direction: column;
        justify-content: space-around;
        margin-left: 20px;
    }

    /* Individual label styling */
    .label {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
    }

    .label span {
        font-weight: normal;
        color: #777;
    }

    .label-human {
        color: #65c6ba; /* Green */
    }

    .label-mixed {
        color: #6A5ACD; /* Purple */
    }

    .label-ai {
        color: #0e8c7d; /* Green Dark */
    }

    /* Chart description styling */
    .chart-description {
        margin-top: 15px;
        font-size: 12px;
        color: #777;
        text-align: center;
    }

    /* END Style for CHARTS */

    /* Title Styling */

    .title-big {
    font-size: 36px;
    font-weight: bold;
    color: black;
    text-align: left; /* Align the title to the left */
    margin-bottom: 10px; /* Reduce the space below the title */
    }

    .title-small {
        font-size: 30px;
        font-weight: bold;
        color: black;
        text-align: left; /* Align the title to the left */
        margin-top: 0; /* Remove extra space above */
        margin-bottom: 15px; /* Adjust the space below */
    }

    .title-small span {
        color: #65c6ba; /* Green for 'Creator' */
    }

    /* Paragraph and Content Text Styling */
    .content-text {
        font-size: 20px; /* Adjust font size */
        color: #333; /* Text color */
        margin: 0 20px; /* Add some horizontal margin */
        text-align: left; /* Align text to the left */
        line-height: 1.6; /* Adjust line spacing */
    }

    .paragraph {
        margin-top: 15px; /* Adjust spacing between paragraphs */
        margin-bottom: 15px; /* Adjust spacing between paragraphs */
    }

    /* Additional customizations to ensure responsiveness and alignment */
    @media (max-width: 768px) {
        .title-big, .title-small, .content-text, .paragraph {
            text-align: left;
            margin-left: 15px; /* Adjust left margin for smaller screens */
        }
    }
    /* END Style for title */


    .logo_image {
        margin-top: 20px;
        text-align: center;
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
    }

    .content-text {
        font-size: 20px;  /* Increased font size */
        color: #333;
        margin: 20px;
    }

    .paragraph {
        font-size: 20px;  /* Increase font size for paragraphs */
        margin-top: 20px;
        margin-bottom: 15px; /* You had this but now explicitly placed it here */
    }

    /* Scan for AI and Clear Buttons */
    div.stButton > button:first-child {
        background-color: #65c6ba; /* Base color */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }

    div.stButton > button:first-child:hover {
        background-color: #0e8c7d; /* Slightly darker shade on hover */
    }

    /* Align Clear button to the right */
    .stClearContainer {
        display: flex;
        justify-content: flex-end; /* Push the Clear button to the right */
        margin-top: -20px; /* Adjust spacing */
    }

    div.stButton > button:last-child {
        background-color: #65c6ba; /* Base color for Clear button */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }

    div.stButton > button:last-child:hover {
        background-color: #0e8c7d; /* Slightly darker shade on hover */
    }

    </style>
""", unsafe_allow_html=True)



# Header

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Encode all the images: (List of all images will be here)
parent_path = pathlib.Path(__file__).parent.resolve()
save_path = os.path.join(parent_path, "data")

logo_base64_logo = get_base64_image(f"{save_path}/logo3_transparent.png")
logo_base64_youtube = get_base64_image(f"{save_path}/youtube.jpg")

# Embed the Base64 image into the HTML
st.markdown(
    f"""
    <div class="header">
        <div class="logo">
            <img src="data:image/png;base64,{logo_base64_logo}" alt="TrueNet Logo">
        </div>
        <div class="nav-links">
            <a href="page_approach">Approach</a>
            <a href="page_models">Models</a>
            <a href="page_resources">Resources</a>
            <a href="page_news">News</a>
            <a href="page_team">Team</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# define session_state variables
if 'text_input' not in st.session_state:
    st.session_state.text_input = f"This is example of text for"

def display_results(analysis: dict):
    prediction_class = analysis["prediction"]
    prediction_proba = analysis["predict_proba"]
    prediction_proba_cleaned = float(prediction_proba.replace('%', ''))

    proba_what = "Human"
    if prediction_class == 0:
        proba_what = "Human"
    else:
        proba_what = "AI"

    proba_human = 0.0
    proba_ai = 0.0
    if prediction_class == 0:
        proba_human = prediction_proba_cleaned
        proba_ai = 100 - proba_human
    else:
        proba_ai = prediction_proba_cleaned
        proba_human = 100 - proba_ai

    # dynamic styling
    ring_chart_gradient_css = f"""background: conic-gradient(
            #65c6ba 0% {proba_human}%, /* Dark green for Human */
            #0e8c7d {proba_human}% 100% /* Gold for AI */
        );
    """

    # HTML for the chart
    st.markdown(
        f"""
        <div class="chart-container">
            <!-- Ring Chart -->
            <div class="ring-chart" style="{ring_chart_gradient_css}">
                <div class="chart-label">
                    {prediction_proba_cleaned}%<br><span style="color: #777; font-size: 12px;">{proba_what}</span>
                </div>
            </div>
            <!-- Labels -->
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
        f'{BASEURL}/text_single_predict',
        headers=headers, params=params
    )
    st.success("Prediction done ✅")
    return response.json()

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
        key="pills_selection"
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

# Divider
st.markdown("---")

# Left and Right Columns
col1, col2 = st.columns([1.5, 2])

# Left Side Content
with col1:
    # Titles
    st.markdown('<div class="title-big">Beyond the Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-small">Identify the <span>Creator</span></div>', unsafe_allow_html=True)

    # Content Text
    st.markdown("""
    <div class="content-text">
        <div class="paragraph">
        In a world of exponential AI-generated content, distinguishing genuine human effort is more critical than ever. TrueNet goes beyond detection—it's your partner for transparency, integrity, and productivity.
        </div>
        <div class="paragraph">
        With advanced AI algorithms and intuitive tools, TrueNet empowers businesses, educators, and creators to protect what’s genuine while embracing responsible AI use.
        </div>
        <div class="paragraph">
        TrueNet<span style="color:#65c6ba;">: Because <span style="color:#0e8c7d;">truth</span> matters.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Insert video (Check issue with encoding)
   # st.video("data/dummy_video.mp4")


# Insert Base64-encoded image inside a styled div
    st.markdown(
        f"""
        <div class="logo_image_manu">
            <img src="data:image/png;base64,{logo_base64_youtube}" alt="Your Image" style="max-width: 100%; border-radius: 8px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# Add a signup button below the image
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <button style="
                background-color: #65c6ba;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                transition: 0.3s;
            "
            onmouseover="this.style.backgroundColor='#0e8c7d'"
            onmouseout="this.style.backgroundColor='#65c6ba'">
                Sign Up
            </button>
        </div>
        """,
        unsafe_allow_html=True
    )


# Right Side Content with Ring Chart and Labels
with col2:
    st.subheader("AI or Human?")

    # st.radio("Try an example:", ["ChatGPT", "Claude", "Human", "AI + Human"], horizontal=True)
    example_buttons()

    text = st.text_area("Your text here",
                 key="text_input",
                 height=200,
                 help="Enter the text you want to analyze")

    # st.file_uploader("Upload File", type=["txt", "docx", "pdf"], key="uploader")

    # Scan and Clear Buttons
    col1, col2, col3 = st.columns([2, 5, 1])  # Adjust column ratios for spacing

    analysis = None
    # Scan for AI Button
    with col1:
        if st.button("Scan for AI", type="primary"):
            print(f'st.session_state: {st.session_state}')

            if len(st.session_state.text_input) > 1:
                with st.spinner('Wait for it...'):
                    analysis = analyze_text(text)
            else:
                st.warning("Please enter some text to analyze.")

    # Empty middle column for spacing
    with col2:
        st.markdown("")

    # Clear Button
    with col3:
        if st.button("Clear", type="secondary"):
            st.session_state.clear()
            st.session_state.text_input = ''
            st.session_state.pills_selection = None
            st.rerun()

    if analysis is not None:
        display_results(analysis)
    # st.markdown('<button class="scan-button">Scan for AI</button>', unsafe_allow_html=True)


# Add a footer at the end of the page
st.markdown(
    """
    <footer style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f5f5f5; padding: 10px 0; border-top: 1px solid #ddd; text-align: center;">
        <a href="#" style="margin-right: 15px; text-decoration: none; color: #65c6ba; font-weight: bold; transition: color 0.3s;"
           onmouseover="this.style.color='#0e8c7d';" onmouseout="this.style.color='#65c6ba';">Contact</a>
        <a href="#" style="margin-right: 15px; text-decoration: none; color: #65c6ba; font-weight: bold; transition: color 0.3s;"
           onmouseover="this.style.color='#0e8c7d';" onmouseout="this.style.color='#65c6ba';">Support</a>
        <a href="#" style="text-decoration: none; color: #65c6ba; font-weight: bold; transition: color 0.3s;"
           onmouseover="this.style.color='#0e8c7d';" onmouseout="this.style.color='#65c6ba';">Imprint</a>
        <p style="margin-top: 5px; font-size: 12px; color: #777;">© 2024 TrueNet</p>
    </footer>
    """,
    unsafe_allow_html=True
)

# hack to speed up the 1st prediction request
import requests
requests.get(f'{BASEURL}/ping')
