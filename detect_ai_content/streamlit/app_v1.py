import streamlit as st
import requests
import base64
import pathlib
import os

# Page Configuration
st.set_page_config(page_title="TrueNet - AI Detection", layout="wide")

# Helper Function: Convert image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Load logo and other images
parent_path = pathlib.Path(__file__).parent.resolve()
save_path = os.path.join(parent_path, "data")
logo_base64_logo = get_base64_image(f"{save_path}/logo3_transparent.png")
logo_base64_youtube = get_base64_image(f"{save_path}/youtube.jpg")

# CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }
    /* Header Styling */
    .header {
        display: flex;
        align-items:;
        justify-content: space-between;
        background-color: #f5f5f5;
        padding: 10px 20px;
        border-bottom: 1px solid #ddd;
    }
    .header .logo img {
        height: 60px;
    }
    .nav-links a {
        text-decoration: none;
        color: #65c6ba;
        font-weight: bold;
        margin-right: 15px;
    }
    .nav-links a:hover {
        color: #0e8c7d;
    }
    .btn-dark {
        background-color: #65c6ba;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        font-weight: bold;
    }
    .btn-dark:hover {
        background-color: #0e8c7d;
    }
    .title-big {
        font-size: 36px;
        font-weight: bold;
        color: black;
    }
    .title-small span {
        color: #65c6ba;
    }
    .scan-button {
        background-color: #65c6ba;
        color: white;
        font-size: 18px;
        padding: 15px 40px;
        border-radius: 30px;
        border: none;
        cursor: pointer;
    }
    .scan-button:hover {
        background-color: #0e8c7d;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
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
        <div>
            <div class="btn-dark">Dashboard</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Divider
st.markdown("---")

# Content Section
col1, col2 = st.columns([1.5, 2])

# Left Content: Title, Description, Video/Image, Signup Button
with col1:
    st.markdown('<div class="title-big">Beyond the Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-small">Identify the <span>Creator</span></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div>
            Since inventing AI detection, TrueNet incorporates the latest research in detecting ChatGPT, GPT4, Google-Gemini, LLaMa, and new AI models, and investigating their sources.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div>
            <img src="data:image/png;base64,{logo_base64_youtube}" alt="YouTube Video" style="max-width: 100%; border-radius: 8px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <button class="scan-button">Sign Up</button>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Right Content: Text Input, Upload, Buttons, and Results
with col2:
    st.subheader("AI or Human?")
    st.write("Try an example:")
    example_cols = st.columns(4)

    # Example Buttons
    examples = [("Llama2", "llama2_chat"), ("Claude", "darragh_claude_v6"),
                ("ChatGPT", "chat_gpt_moth"), ("Human", "persuade_corpus")]
    for i, (label, source) in enumerate(examples):
        if example_cols[i].button(label):
            response = requests.get(f'https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/random_text?source={source}')
            st.session_state.text_input = response.json().get('text', '')

    # Text Input
    text = st.text_area("Your text here", value=st.session_state.text_input, height=200)

    # File Upload
    st.file_uploader("Upload File", type=["txt", "docx", "pdf"])

    # Buttons: Scan and Clear
    scan_col, clear_col = st.columns([2, 1])
    with scan_col:
        if st.button("Scan for AI"):
            if text.strip():
                with st.spinner("Analyzing..."):
                    headers = {"accept": "application/json"}
                    params = {"text": text}
                    response = requests.get("https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/text_single_predict",
                                            headers=headers, params=params)
                    # Process and display results
                    st.write("Prediction done!")
                    st.metric("Confidence", "100% Human")
            else:
                st.warning("Please enter text to analyze.")
    with clear_col:
        if st.button("Clear"):
            st.session_state.text_input = ""

# Footer
st.markdown(
    """
    <footer style="text-align: center; margin-top: 50px; font-size: 14px; color: #777;">
        <a href="#" style="margin-right: 15px; color: #65c6ba; font-weight: bold;">Contact</a>
        <a href="#" style="margin-right: 15px; color: #65c6ba; font-weight: bold;">Support</a>
        <a href="#" style="color: #65c6ba; font-weight: bold;">Imprint</a>
        <p>© 2024 TrueNet</p>
    </footer>
    """,
    unsafe_allow_html=True,
)

# Warmup for Backend
requests.get('https://detect-ai-content-improved14nov-667980218208.europe-west1.run.app/ping')
