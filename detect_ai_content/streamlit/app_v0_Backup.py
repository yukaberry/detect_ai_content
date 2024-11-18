import streamlit as st

# Page Configuration
st.set_page_config(page_title="TrueNet - AI Detection", layout="wide")

# CSS for Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f7f7;
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        border-bottom: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    .header img {
        max-height: 50px;
    }
    .nav-links {
        display: flex;
        gap: 10px;  /* Reduced gap */
        font-size: 16px;
        font-weight: bold;
        color: #333333;
        padding-left: 0;  /* Aligned with logo */
    }
    .nav-links a {
        text-decoration: none;
        color: #333333;
    }
    .nav-links a:hover {
        color: #28a745;
    }
    .btn-outline, .btn-dark {
        padding: 8px 16px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        margin-right: 10px;  /* Space between buttons */
    }
    .btn-outline {
        border: 1px solid #333333;
        color: #333333;
        background: transparent;
    }
    .btn-outline:hover {
        background-color: #333333;
        color: white;
    }
    .btn-dark {
        background-color: #333333;
        color: white;
    }
    .btn-dark:hover {
        background-color: #555555;
    }
    .title-big, .title-small {
        font-size: 36px;
        font-weight: bold;
        color: black;
    }
    .title-small span {
        color: #28a745;  /* Green for 'Creator' */
    }
    .paragraph {
        font-size: 18px;
        margin-top: 20px;
    }
    .scan-button {
        background-color: black;
        color: white;
        font-size: 18px;
        padding: 15px 40px;
        border-radius: 30px;
        border: none;
        cursor: pointer;
        margin-left: 0;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section

st.markdown("""
    <div class="header">

<div id"logo"></div>

        <div class="nav-links">
            <a href="#">Approach</a>
            <a href="#">Models</a>
            <a href="#">Resources</a>
            <a href="#">News</a>
            <a href="#">Team</a>
        </div>
        <div class="button-container">
            <div class="btn-dark">Dashboard
        </div>


    </div>
""", unsafe_allow_html=True)

# Insert Image with API
# Insert image using st.image


# Divider
st.markdown("---")

# Layout
col1, col2 = st.columns([1.5, 2])

# Left Side Content
with col1:
    st.markdown('<div class="title-big">Beyond the Surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-small">Identify the <span>Creator</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="paragraph">Since inventing AI detection, TrueNet incorporates the latest research in detecting ChatGPT, GPT4, Google-Gemini, LLaMa, and new AI models, and investigating their sources.</div>', unsafe_allow_html=True)

# Right Side Content
with col2:
    st.subheader("AI or Human?")
    st.radio("Try an example:", ["ChatGPT", "Claude", "Human", "AI + Human"], horizontal=True)
    st.text_area("Your text here", placeholder="Your text here...")
    st.file_uploader("Upload File", type=["txt", "docx", "pdf"], key="uploader")
    st.markdown('<button class="scan-button">Scan for AI</button>', unsafe_allow_html=True)
