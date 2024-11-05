import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detect_ai_content.ml_logic.preprocess import preprocess_text
from detect_ai_content.ml_logic.for_images.vgg16_improved import clean_img_vgg16
from detect_ai_content.ml_logic.for_images.cnn import load_cnn_model, clean_img_cnn
from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import load_model

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

@app.post("/image_predict")
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
    img = clean_img_vgg16(img_data)

    # Predict
    predicted_class = app.state.model.predict(img)

    # Get predicted indices
    predicted_probabilities = np.argmax(predicted_class, axis=1)

    # TODO
    # errors below but prediction and return predicted value worked!

    # # Get class labels
    # train_images = retrain_images()
    # labels = getattr(train_images, 'class_indices', None)
    # # Debug output to check labels content
    # print(Fore.BLUE + f"\nLabels loaded!" + Style.RESET_ALL)

    # if labels:
    #     labels = {v: k for k, v in labels.items()}
    #     print(Fore.BLUE + f"{labels}" + Style.RESET_ALL)
    # else:
    #     print(Fore.RED + "\nError: No class labels found in train_images.class_indices" + Style.RESET_ALL)
    #     return {"error": "No class labels found in the model. Check dataset and retrain_images function."}
    # # Convert indices back to class names
    # try:
    #     prediction = [labels.get(int(k), "Unknown") for k in predicted_probabilities]
    # except KeyError as e:
    #     print(Fore.RED + f"\nKeyError: {e} - predicted_probabilities contains unknown label indices." + Style.RESET_ALL)
    #     return {"error": "Prediction contains unknown label indices. Check class indices and labels dictionary."}
    # print(Fore.BLUE + f"\nPrediction complete!" + Style.RESET_ALL)
    # print(Fore.BLUE + f"{prediction}" + Style.RESET_ALL)

    #{"prediction": int(predicted_probabilities)}
    #{"prediction": int(prediction)}


    # if predicted_probabilities == 1:
    #     prediction_message = "Predicted as AI"
    # elif predicted_probabilities == 0:
    #     prediction_message = "Predicted as Human"

    return {"prediction": int(predicted_probabilities)}
    # return {"prediction": int(predicted_probabilities),
    #          "message": prediction_message}


@app.post("/image_predict_cnn")
async def predict(img: UploadFile = File(...)):

    """
    - return a single prediction.
    - 'img'(RGB) is provided
    - 0 likely representing 'FAKE' and 1 representing 'REAL'.
    """

    if app.state.model is None:
        app.state.model = load_cnn_model()

    # Load image  # Read image data asynchronously
    img_data = await img.read()

    # Clean/reshape user-input image
    img = clean_img_cnn(img_data)

    # Predict
    predicted_class = app.state.model.predict(img)

    # Get predicted indices
    predicted_probabilities = np.argmax(predicted_class, axis=1)

    # prediction message
    # 0 likely representing 'FAKE' and 1 representing 'REAL'
    if predicted_probabilities == 0:
        prediction_message = "Predicted as AI"
    elif predicted_probabilities == 1:
        prediction_message = "Predicted as Human"


    return {"prediction": int(predicted_probabilities),
             "message": prediction_message}


@app.get("/")
def root():
    return {
        'greeting': 'Hello Detect AI Content !! '
    }
