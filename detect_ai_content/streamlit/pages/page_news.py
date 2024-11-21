# pages/page_news.py
import streamlit as st
import base64
import os
import pathlib

# Convert the image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def app():
    # Prepare base64 string for the logo
    parent_path = pathlib.Path(__file__).parent.parent.resolve()
    save_path = os.path.join(parent_path, "data")
    logo_base64 = get_base64_image(f"{save_path}/logo3_transparent.png")

    # Display custom header with logo
    st.markdown(f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:10px;">
            <img src="data:image/png;base64,{logo_base64}" alt="TrueNet Logo" style="height:60px; width:auto;">

        </div>
    """, unsafe_allow_html=True)

    # Main content of the page
    st.write("""
    ## Catch up on the latest updates and developments

    - ** place some links here
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

app()
