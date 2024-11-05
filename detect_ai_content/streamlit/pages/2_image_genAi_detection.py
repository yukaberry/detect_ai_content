import streamlit as st
from PIL import Image
import numpy as np
import os
import pathlib
import io
import requests
import time

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
    os.makedirs(save_path, exist_ok=True)
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
                'img': ('image.jpg', open(complete_name, 'rb'), 'image/jpg'),
            }
            # local http://0.0.0.0:8000/image_predict
            #'https://detect-ai-content-j-mvp-667980218208.europe-west1.run.app/image_predict', headers=headers, files=files)
            response = requests.post("https://detect-ai-content-667980218208.europe-west1.run.app/image_predict",headers=headers, files=files)
            st.success(f"{response.json()}")
