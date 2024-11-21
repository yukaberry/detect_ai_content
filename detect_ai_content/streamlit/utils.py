import streamlit as st

def configure_sidebar():
    pages = {
        "App Navigation": [
            st.Page("./app_v0.py", title="Home"),
        ],
        "📖 AI detection for texts": [
            st.Page("./pages/1_Text AI_Detector.py", title="One model"),
            st.Page("./pages/2_Multi-Model_Text_AI_Detector.py", title="Multiple models"),
        ],
        "🌌 AI detection for images": [
            st.Page("./pages/3_Image_AI_Detector.py", title="One model"),
            st.Page("./pages/4_Multi-Model_Image_AI_Detector.py", title="Multiple models"),
        ],
    }
    pg = st.navigation(pages)
    pg.run()
