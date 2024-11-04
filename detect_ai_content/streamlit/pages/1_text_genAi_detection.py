import streamlit as st
import requests
import time

st.subheader("Text to analyze")

txt = st.text_area(
    " ",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",
)

st.write(f"You wrote {len(txt)} characters.")
if st.button('Evaluate this text'):
    st.subheader("Prediction - predict")
    with st.spinner('Wait for it...'):
        time.sleep(2)
        headers = {
            'accept': 'application/json',
        }
        params = {
            "text":txt
        }
        response = requests.get('https://detect-ai-content-j-mvp-667980218208.europe-west1.run.app/predict', headers=headers, params=params)
        st.success(f"{response.json()}")
