import streamlit as st

'''
# Detect AI Content front
'''

import streamlit as st


txt = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",
)

# st.write(f"You wrote {len(txt)} characters.")

# st.subheader("Prediction - get text")
st.write(f"{txt}")

st.subheader("Prediction - preprocess")
# X_processed = preprocess_text(txt)
# st.dataframe(X_processed)

st.subheader("Prediction - predict")
# y_pred = model.predict(X_processed)
# st.write(f"{y_pred}")
