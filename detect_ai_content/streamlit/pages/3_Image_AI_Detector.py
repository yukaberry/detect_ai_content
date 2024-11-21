import streamlit as st
from PIL import Image
import numpy as np
import os
import pathlib
import io
import requests
import time
from params import *
import base64

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

    # Title
    st.markdown("""
        <h2 style='text-align: left; color: black;
        font-size: 30px'>Is your image Real or AI Generated?</h2>

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.9rem;
            color: #777777;

    """, unsafe_allow_html=True)




    # https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
    # SUPPORT PNG in the futur
    # Complex to support the 4th layer on the Api & model
    uploaded_file = st.file_uploader("Upload an image", type=["jpg"], accept_multiple_files=False)

    if uploaded_file is not None:
        path_in = uploaded_file.name
        print(path_in)
    else:
        path_in = None

    if path_in is not None:

        st.subheader("Your image")
        img_array = np.array(uploaded_file)
        st.image(
            uploaded_file,
            caption=f"You amazing image has shape {img_array.shape[0:2]}",
            use_column_width=True,
        )

        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))

        parent_path = pathlib.Path(__file__).parent.parent.resolve()
        save_path = os.path.join(parent_path, "data")

        # Create the directory if it doesn't exist
        #os.makedirs(save_path, exist_ok=True)
        #

        complete_name = os.path.join(save_path, uploaded_file.name)

        with open(complete_name, "wb") as f:
            f.write(image_bytes)

        st.subheader("Image saved at ")
        st.write(f"{complete_name}")

        if st.button('Evaluate this image'):
            st.subheader("Prediction - predict")
            with st.spinner('Wait for it...'):
                time.sleep(2)

                headers = {
                    'accept': 'application/json',
                    # requests won't add a boundary if this header is set when you pass files=
                    # 'Content-Type': 'multipart/form-data',
                }

                files = {
                    'file': (uploaded_file.name, open(complete_name, 'rb'), 'image/jpg'),

                }
                # local test
                # http://0.0.0.0:8000/image_predict_most_updated_cnn
                # deploy
                #'https://detect-ai-content-j-mvp-667980218208.europe-west1.run.app/image_predict_most_updated_cnn', headers=headers, files=files)

                # final mode from  Ab
                import requests
                response = requests.post(f"{BASEURL}/image_predict_most_updated_cnn",headers=headers, files=files)

                st.success(response.text)



    # Add the "Clear Content" button
    if st.button('Clear Content'):
        st.session_state.uploaded_file = None  # Reset session state
        st.markdown(" ###Please upload a new file.")

    # Ping server to preload things if needed
    import requests
    requests.get(f'{BASEURL}/ping')



   # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center;">
            © 2024 TrueNet AI Detection. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

app()
