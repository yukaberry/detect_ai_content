import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from detect_ai_content.ml_logic.registry import load_model
from detect_ai_content.ml_logic.preprocess import preprocess_text

app = FastAPI()
app.state.model = None

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?text=lsjisefohlksdjf
@app.get("/predict")
def predict(
        text: str
    ):
    """
    Make a single prediction prediction.
    Assumes `text` is provided as a string
    """

    if app.state.model is None:
        app.state.model = load_model()

    X_processed = preprocess_text(text)
    y_pred = app.state.model.predict(X_processed)
    print(f"one pred: {y_pred[0]}")
    return {
        'prediction': int(y_pred[0])
    }

@app.get("/")
def root():
    return {
        'greeting': 'Hello Detect AI Content '
    }
