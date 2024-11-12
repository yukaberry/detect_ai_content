import pandas as pd
import numpy as np
import mlflow

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import TrueNetTextLogisticRegression
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import TrueNetTextDecisionTreeClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import TrueNetTextKNeighborsClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextRNN import TrueNetTextRNN
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import TrueNetTextSVC
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import TrueNetTextTfidfNaiveBayesClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import TrueNetTextUsingBERTMaskedPredictions


from detect_ai_content.ml_logic.preprocess import preprocess
from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions

from detect_ai_content.ml_logic.for_images.vgg16_improved import load_model_vgg16
from detect_ai_content.ml_logic.for_images.vgg16_improved import clean_img_vgg16
from detect_ai_content.ml_logic.for_images.cnn import load_cnn_model, clean_img_cnn

app = FastAPI()
app.state.model_text = None
app.state.model_image = None
app.state.model_image_cnn = None

app.state.models = {}

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

    if app.state.model_text is None:
        app.state.model = TrueNetTextLogisticRegression().model

    text_df = pd.DataFrame(data=[text],columns=['text'])
    X_processed = preprocess(text_df)
    y_pred = app.state.model_text.predict(X_processed)

    print(f"one pred: {y_pred[0]}")
    return {
        'prediction': int(y_pred[0])
    }

@app.get("/text_multi_predict")
def predict(
        text: str
    ):
    """
    Make a single prediction prediction.
    Assumes `text` is provided as a string
    """

    predictions = {}

    text_df = pd.DataFrame(data=[text],columns=['text'])
    text_enriched_df = enrich_text(text_df)
    text_enriched_df = enrich_text_BERT_predictions(text_enriched_df)

    # Using "classic" preprocessed data
    X_processed = preprocess(text_enriched_df, auto_enrich=False)

    if "TrueNetTextLogisticRegression" not in app.state.models:
        app.state.models["TrueNetTextLogisticRegression"] = TrueNetTextLogisticRegression()._load_model(stage="staging")
    model = app.state.models["TrueNetTextLogisticRegression"]
    y_pred = model.predict(X_processed)
    predictions["TrueNetTextLogisticRegression"] = int(y_pred[0])

    if "TrueNetTextDecisionTreeClassifier" not in app.state.models:
        app.state.models["TrueNetTextDecisionTreeClassifier"] = TrueNetTextDecisionTreeClassifier()._load_model(stage="staging")
    model = app.state.models["TrueNetTextDecisionTreeClassifier"]
    y_pred = model.predict(X_processed)
    predictions["TrueNetTextDecisionTreeClassifier"] = int(y_pred[0])

    if "TrueNetTextKNeighborsClassifier" not in app.state.models:
        app.state.models["TrueNetTextKNeighborsClassifier"] = TrueNetTextKNeighborsClassifier()._load_model(stage="staging")
    model = app.state.models["TrueNetTextKNeighborsClassifier"]
    y_pred = model.predict(X_processed)
    predictions["TrueNetTextKNeighborsClassifier"] = int(y_pred[0])

    # if "TrueNetTextRNN" not in app.state.models:
    #    app.state.models["TrueNetTextRNN"] = TrueNetTextRNN()._load_model(stage="staging")
    # model = app.state.models["TrueNetTextRNN"]
    # y_pred = model.predict(X_processed)
    # predictions["TrueNetTextRNN"] = y_pred

    # Using only "BERT" preprocess

    if "TrueNetTextUsingBERTMaskedPredictions" not in app.state.models:
        app.state.models["TrueNetTextUsingBERTMaskedPredictions"] = TrueNetTextUsingBERTMaskedPredictions()._load_model(stage="staging")
    model = app.state.models["TrueNetTextUsingBERTMaskedPredictions"]
    y_pred = model.predict(TrueNetTextUsingBERTMaskedPredictions.preprocess(data=text_enriched_df))
    predictions["TrueNetTextUsingBERTMaskedPredictions"] = int(y_pred[0])

    # Using raw data

    if "TrueNetTextTfidfNaiveBayesClassifier" not in app.state.models:
        app.state.models["TrueNetTextTfidfNaiveBayesClassifier"] = TrueNetTextTfidfNaiveBayesClassifier()._load_model(stage="staging")
    model = app.state.models["TrueNetTextTfidfNaiveBayesClassifier"]
    y_pred = model.predict(text_df)
    predictions["TrueNetTextTfidfNaiveBayesClassifier"] = int(y_pred[0])

    print(f"preds: {predictions}")
    return {
        'predictions': predictions
    }

@app.post("/image_predict")
async def predict(img: UploadFile = File(...)):
    """
    Make a single prediction prediction.
    Assumes `img` is provided
    """
    if app.state.model_image is None:
        app.state.model_image = load_model_vgg16()

    # Load image  # Read image data asynchronously
    img_data = await img.read()

    # Clean/reshape user-input image
    img = clean_img_vgg16(img_data)

    # Predict
    predicted_class = app.state.model_image.predict(img)

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

    if app.state.model_image_cnn is None:
        app.state.model_image_cnn = load_cnn_model()

    # Load image  # Read image data asynchronously
    img_data = await img.read()

    # Clean/reshape user-input image
    img = clean_img_cnn(img_data)

    # Predict
    predicted_class = app.state.model_image_cnn.predict(img)

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
        'greeting': 'Hello Detect AI Content !! hello '
    }
