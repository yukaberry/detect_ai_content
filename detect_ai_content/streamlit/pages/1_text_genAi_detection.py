import streamlit as st
from typing import Optional
import requests

def set_page_config():
    st.set_page_config(
        page_title="AI Text Analyzer",
        page_icon="🔍",
        layout="wide"  # Changed to wide to accommodate the menu
    )

def local_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        /* Navigation Styles */
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: white;
            border-bottom: 1px solid #f0f0f0;
            margin-bottom: 2rem;
        }
        .nav-logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            text-decoration: none;
            display: flex;
            align-items: center;
        }
        .nav-logo img {
            height: 30px;
            margin-right: 10px;
        }
        .nav-menu {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        .nav-item {
            color: #333;
            text-decoration: none;
            font-weight: 500;
        }
        .nav-buttons {
            display: flex;
            gap: 1rem;
        }
        .demo-button {
            background: white;
            border: 1px solid #333;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            color: #333;
        }
        .dashboard-button {
            background: #333;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            color: white;
        }
        /* Content Styles */
        .stButton>button {
            width: 100%;
            border-radius: 4px;
            padding: 0.5rem;
        }
        .text-input {
            min-height: 200px;
        }
        .example-btn {
            margin: 0.25rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            border: 1px solid #e0e0e0;
            background: white;
            cursor: pointer;
        }
        /* Hide Streamlit's default menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def create_navigation():
    nav_html = """
        <div class="nav-container">
            <a href="#" class="nav-logo">
                <span>AI Text Analyzer</span>
            </a>
            <div class="nav-menu">
                <div class="dropdown">
                    <a href="#" class="nav-item">Products</a>
                </div>
                <div class="dropdown">
                    <a href="#" class="nav-item">Solutions</a>
                </div>
                <div class="dropdown">
                    <a href="#" class="nav-item">Resources</a>
                </div>
                <a href="#" class="nav-item">Pricing</a>
                <a href="#" class="nav-item">News</a>
                <a href="#" class="nav-item">Team</a>
            </div>
            <div class="nav-buttons">
                <button class="demo-button">Request Demo</button>
                <button class="dashboard-button">Dashboard</button>
            </div>
        </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)

def example_buttons():
    st.write("Try an example:")

    # Create a container for the buttons
    button_container = st.container()

    # Use columns within the container
    cols = button_container.columns(4)

    # Track which button was clicked
    selected_example = None

    # Create all buttons and check their states
    if cols[0].button("ChatGPT", key="example1", type="secondary"):
        selected_example = "example_chatgpt_text"

    if cols[1].button("Claude", key="example2", type="secondary"):
        selected_example = "example_claude_text"

    if cols[2].button("Human", key="example3", type="secondary"):
        selected_example = "example_human_text"

    if cols[3].button("AI + Human", key="example4", type="secondary"):
        selected_example = "example_hybrid_text"

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
    response = requests.get('https://detect-ai-content-667980218208.europe-west1.run.app/predict', headers=headers, params=params)
    st.success(response.text)
    return response.json()

def display_results(analysis: dict):
    st.markdown("### Analysis Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", analysis["prediction"])
    with col2:
        st.metric("Confidence", f"{analysis['confidence']:.1%}")

    st.markdown("#### Detailed Metrics")
    metrics = analysis["details"]
    cols = st.columns(len(metrics))
    for col, (metric, value) in zip(cols, metrics.items()):
        col.metric(
            metric.replace("_", " ").title(),
            f"{value:.1%}"
        )

def create_content():
    st.title("More than an AI detector.")
    st.markdown("### Preserve what's human.")
    st.write("Since inventing AI detection, we incorporate the latest research in detecting ChatGPT, GPT4, Google-Gemini, LLaMa, and new AI models, and investigating their sources.")

    # Store the selected example in session state if it's not already there
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None

    # Get newly selected example
    new_selection = example_buttons()

    # Update session state if a new selection was made
    if new_selection is not None:
        st.session_state.selected_example = new_selection
        st.session_state.text_input = f"This is example text for {new_selection}"

    text = text_input_section()

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Analyze Text", type="primary"):
            if text:
                with st.spinner('Wait for it...'):
                    analysis = analyze_text(text)
                    display_results(analysis)
            else:
                st.warning("Please enter some text to analyze.")
    with col2:
        if st.button("Clear", type="secondary"):
            st.session_state.clear()
            st.rerun()


def main():
    set_page_config()
    local_css()

    # Add navigation
    # create_navigation()

    # Create main content in a container
    with st.container():
        create_content()

if __name__ == "__main__":
    main()
