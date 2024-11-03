import streamlit as st
import requests

txt = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",
)

st.write(f"You wrote {len(txt)} characters.")

st.subheader("Prediction - get text")
st.write(f"{txt}")

st.subheader("Prediction - predict")
headers = {
    'accept': 'application/json',
}
params = {
    "text":txt
}
response = requests.get('https://detect-ai-content-j-mvp-667980218208.europe-west1.run.app/predict', headers=headers, params=params)
st.write(f"{response.json()}")
