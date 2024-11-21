
import streamlit as st

pages = {
    "App Navigation": [
        st.Page("./app_v0.py", title="Home"),
    ],
    "📖 AI detection for texts": [
        st.Page("./pages/1_Text AI_Detector.py", title="Predict using the best model"),
        st.Page("./pages/2_Multi-Model_Text_AI_Detector.py", title="Challenge our models"),
    ],
    "🌌 AI detection for images": [
        st.Page("./pages/3_Image_AI_Detector.py", title="Predict using the best model"),
        st.Page("./pages/page_multi_imges.py", title="Challenge our models"),
    ],
    "ℹ️ More pages ": [
        st.Page("./pages/page_approach.py", title="Approach"),
        st.Page("./pages/page_models.py", title="Models"),
        st.Page("./pages/page_news.py", title="News"),
        st.Page("./pages/page_resources.py", title="Resources"),
        st.Page("./pages/page_team.py", title="Team"),
        # st.page_link("https://github.com/yukaberry/detect_ai_content", label="Github repo")
    ],
}

pg = st.navigation(pages)
pg.run()

st.sidebar.page_link(
    page="https://github.com/yukaberry/detect_ai_content",
    label="Github repo "
)
