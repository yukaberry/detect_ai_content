import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from detect_ai_content.utils import load_model
from detect_ai_content.utils import clean_img
from colorama import Fore, Style

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
@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    """
    Make a single prediction prediction.
    Assumes `img` is provided
    """
    if app.state.model is None:
        app.state.model = load_model()

    # Load image  # Read image data asynchronously
    img_data = await img.read()

    # Clean/reshape user-input image
    img = clean_img(img_data)

    # Predict
    predicted_class = app.state.model.predict(img)

    # Get predicted indices
    predicted_probabilities = np.argmax(predicted_class, axis=1)

    return {"prediction": int(predicted_probabilities)}

@app.get("/")
def root():
    return {'greeting': 'Hello Detect AI Content (img model/CNN)'}
