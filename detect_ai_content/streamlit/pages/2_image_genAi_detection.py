import streamlit as st
import base64
from PIL import Image
import numpy as np
from io import BytesIO

# https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
img_file_buffer = st.file_uploader("Upload an image", type=["png"], accept_multiple_files=False)
image = Image.open(img_file_buffer)

if image is not None:

    st.subheader("Your image")
    img_array = np.array(image)
    st.image(
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}",
        use_column_width=True,
    )

    st.subheader("Your base64 image")
    # base64_document = base64.b64encode(image.read()).decode('utf-8')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    st.write(f"{img_str}")
