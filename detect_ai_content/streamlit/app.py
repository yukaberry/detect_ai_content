import streamlit as st
import sklearn

'''
# Detect AI Content front
'''

import datetime
import streamlit as st
import pickle

from detect_ai_content.ml_logic.preprocess import preprocess_text


st.subheader("🔑 Key concepts")
st.write(f"Explain the basis of the projet ")

st.subheader("📚 Datasets")
st.write(f"Link the datasets")

st.subheader("🧠 Preprocess")
st.write(f"Explain the processes ")

st.subheader("💯 Results & metrics")
st.write(f"What are our results ?")
