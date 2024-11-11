import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import load_model, preprocess

from detect_ai_content.ml_logic.for_images.vgg16 import Vgg16
from detect_ai_content.ml_logic.for_images.TrueNetImageUsinCustomCNN import TrueNetImageUsinCustomCNN

app = FastAPI()
app.state.model_text = None
app.state.model_image = None
app.state.model_image_cnn = None

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
def predict(text: str):

    """
    - make a single prediction prediction using user provided 'text'
    - text : text type only  # TO BE IMPROVED for csv data type etc ...

    """

    if app.state.model_text is None:
        app.state.model_text = load_model()

    text_df = pd.DataFrame(data=[text],columns=['text'])
    X_processed = preprocess(text_df)
    y_pred = app.state.model_text.predict(X_processed)

    if y_pred[0] == 1:
        prediction_message = "Predicted as AI"
    elif y_pred[0] == 0:
        prediction_message = "Predicted as Human"

    return {"prediction": int(y_pred[0]),
             "message": prediction_message}


@app.get("/predict_")
def predict(text: str):

    """
    - make a single prediction prediction using user provided 'text'
    - text : text type only  # TO BE IMPROVED for csv data type etc ...

    """

    if app.state.model_text is None:
        app.state.model_text = load_model()

    text_df = pd.DataFrame(data=[text],columns=['text'])
    X_processed = preprocess(text_df)
    y_pred = app.state.model_text.predict(X_processed)

    if y_pred[0] == 1:
        prediction_message = "Predicted as AI"
    elif y_pred[0] == 0:
        prediction_message = "Predicted as Human"

    return {"prediction": int(y_pred[0]),
             "message": prediction_message}


@app.post("/image_predict_vgg16")
async def predict(user_input: UploadFile = File(...)):

    """
    - make a single prediction prediction using user provided 'img'
    - img : jpg data type only  # TO BE IMPROVED

    """


    # read user input image
    user_input = await user_input.read()
    img = Image.open(BytesIO(user_input))

    # initialise class
    vgg16 = Vgg16()

    # preprocess user input image
    # load model
    # predict
    # return message
    prediction, message = vgg16.predict(img)


    return {"prediction": prediction,
            "message": message}


@app.post("/image_predict_cnn")
async def predict(user_input: UploadFile = File(...)):

    """
    - make a single prediction prediction using user provided 'img'
    - img (RGB) : jpg data type only  # TO BE IMPROVED

    - 0 likely representing 'FAKE' and 1 representing 'REAL'.
    """

    # read user input image
    user_input = await user_input.read()
    img = Image.open(BytesIO(user_input))

    # initialise class
    cnn = TrueNetImageUsinCustomCNN()

    # preprocess user input image
    # load model
    # predict
    # return message
    prediction, message = cnn.predict(img)

    # # 0 likely representing 'FAKE' and 1 representing 'REAL'
    # # TODO label change to avoid confusion

    return {"prediction": prediction,
            "message": message}



@app.get("/")
def root():
    return {'greeting': 'Hello Detect AI Content. imporved : class used'}
