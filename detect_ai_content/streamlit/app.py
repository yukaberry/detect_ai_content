import streamlit as st

'''
# Detect AI Content front
'''

import streamlit as st

st.subheader("🔑 Key concepts")
st.write(f"Explain the basis of the projet ")

st.subheader("📚 Datasets")
st.write(f"present our datasets & Link the datasets")

st.subheader("🧠 Preprocess")
st.write(f"Explain the processes ")

st.subheader("💯 Results & metrics")
st.write(f"What are our results ?")

# Ping server to preload things if needed
import requests
requests.get('https://detect-ai-content-667980218208.europe-west1.run.app/ping')
