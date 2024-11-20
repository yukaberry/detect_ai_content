import streamlit as st
import requests
import pathlib
import os
import base64

# Page Configuration
st.set_page_config(page_title="TrueNet - AI Detection", layout="wide")

# Include CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
        font-family: Arial, sans-serif;
    }

    /* Header styles */
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #f5f5f5;
        padding: 10px 20px;
        border-bottom: 1px solid #ddd; /* Light gray border at the bottom */
    }
    .header img {
        height: 60px;
        width: auto;
    }
    .nav-links {
        display: flex;
        gap: 15px;
    }
    .nav-links a {
    text-decoration: none; /* Remove underline */
    color: #65c6ba; /* Blue color */
    font-weight: bold; /* Bold text */
    transition: color 0.3s; /* Smooth transition for hover effect */
    }
    .nav-links a:hover {
        color: #0e8c7d;
    }

    /* Section titles */
    h1, h2 {
        margin-bottom: 10px;
        line-height: 1.2;
    }
    h2 span {
        color: #65c6ba;
    }

    /* Text styles */
    p {
        font-size: 18px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    /* Button styles */
    button {
        background-color: #65c6ba;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    button:hover {
        background-color: #0e8c7d;
    }

    /* Footer styles */
    footer {
        background-color: #f5f5f5;
        padding: 10px;
        text-align: center;
        font-size: 14px;
        color: #777;
        border-top: 1px solid #ddd;
    }

    /* Chart styles */
    .chart-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    .ring-chart {
        width: 300px;  /* Adjusted size */
        height: 300px;
        border-radius: 50%;
        position: relative;
        background: conic-gradient(#65c6ba 0% 75%, #0e8c7d 75% 100%);
        margin: 20px auto;
    }
    .ring-chart::before {
        content: "";
        position: absolute;
        top: 30px;
        left: 30px;
        width: 240px;
        height: 240px;
        background-color: white;
        border-radius: 50%;
    }
    .chart-label {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 18px;
        font-weight: bold;
        color: #333;
        text-align: center;
    }
    .labels-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px;
    }
    .title-big, .title-small {
        font-size: 36px;
        font-weight: bold;
        color: black;
    }
    .title-small span {
        color: #65c6ba;  /* Green for 'Creator' */
    }
    </style>
""", unsafe_allow_html=True)

# Prepare assets
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path for logo and other images
parent_path = pathlib.Path(__file__).parent.resolve()
save_path = os.path.join(parent_path, "data")
logo_path = os.path.join(save_path, "logo3_transparent.png")

# Header section with logo and navigation
if os.path.exists(logo_path):
    logo_base64 = get_base64_image(logo_path)
    st.markdown(f"""
        <div class="header">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo">
            <div class="nav-links">
                <a href="#">Approach</a>
                <a href="#">Models</a>
                <a href="#">Resources</a>
                <a href="#">News</a>
                <a href="#">Team</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Logo file not found. Please check the path.")

# Divider
st.markdown("---")

# Main content area: two columns
col1, col2 = st.columns([1.5, 2])

# Left column for video and description
with col1:
    st.markdown('<div class="title-big">Beyond the Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-small">Identify the <span>Creator</span></div>', unsafe_allow_html=True)
    st.markdown("""
        <p>In a world of exponential AI-generated content, distinguishing genuine human effort is more critical than ever. TrueNet goes beyond detection—it's your partner for transparency, integrity, and productivity.</p>
        <p>With advanced AI algorithms and intuitive tools, TrueNet empowers businesses, educators, and creators to protect what’s genuine while embracing responsible AI use.</p>
        <p>TrueNet<span style="color:#65c6ba;">: Because <span style="color:#0e8c7d;">truth</span> matters.</span></p>
    """, unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


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


# Right column for AI detection functionality
with col2:
    st.subheader("AI or Human?")
    st.write("Try an example:")
    cols = st.columns(4)
    if cols[0].button("Llama2"):
        st.session_state.text_input = "Example text from Llama2"
    if cols[1].button("Claude"):
        st.session_state.text_input = "Example text from Claude"
    if cols[2].button("ChatGPT"):
        st.session_state.text_input = "Example text from ChatGPT"
    if cols[3].button("Human"):
        st.session_state.text_input = "Example text from Human"

    # Text input and file uploader
    text = st.text_area("Your text here", value=st.session_state.get("text_input", ""), height=200)
    st.file_uploader("Upload File", type=["txt", "docx", "pdf"])

    # Buttons for analysis and clearing input
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Scan for AI", type="primary"):
            if len(st.session_state.text_input) > 1:
                with st.spinner('Wait for it...'):
                    analysis = analyze_text(text)
                    display_results(analysis)
            else:
                st.warning("Please enter some text to analyze.")
    with col2:
        if st.button("Clear", type="secondary"):
            st.session_state.clear()
            st.session_state.text_input = ''
            st.session_state.selected_example = ''
            st.rerun()

# Footer
st.markdown("""
    <footer>
        <a href="#" style="margin-right: 15px;">Contact</a>
        <a href="#" style="margin-right: 15px;">Support</a>
        <a href="#">Imprint</a>
        <p>© 2024 TrueNet</p>
    </footer>
""", unsafe_allow_html=True)
