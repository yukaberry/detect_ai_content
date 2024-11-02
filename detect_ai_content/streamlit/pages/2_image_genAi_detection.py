import streamlit as st
import base64
from PIL import Image
import numpy as np
import os
import pathlib
import io
import requests

# https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
uploaded_file = st.file_uploader("Upload an image", type=["png"], accept_multiple_files=False)

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
    complete_name = os.path.join(save_path, uploaded_file.name)

    with open(complete_name, "wb") as f:
        f.write(image_bytes)

    st.subheader("Image saved at ")
    st.write(f"{complete_name}")

    import requests
    headers = {
        'accept': 'application/json',
        # requests won't add a boundary if this header is set when you pass files=
        # 'Content-Type': 'multipart/form-data',
    }

    files = {
        'img': ('image.png', open(complete_name, 'rb'), 'image/png'),
    }

    response = requests.post('http://127.0.0.1:8000/predict', headers=headers, files=files)
    print(response)
