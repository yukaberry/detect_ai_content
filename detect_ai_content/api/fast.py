import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO

from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import TrueNetTextLogisticRegression
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import TrueNetTextDecisionTreeClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import TrueNetTextKNeighborsClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextRNN import TrueNetTextRNN
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import TrueNetTextSVC
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import TrueNetTextTfidfNaiveBayesClassifier
# from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import TrueNetTextUsingBERTMaskedPredictions


from detect_ai_content.ml_logic.data import enrich_text, enrich_lexical_diversity_readability

from detect_ai_content.ml_logic.for_images.vgg16_improved import load_model_vgg16
from detect_ai_content.ml_logic.for_images.vgg16_improved import clean_img_vgg16
from detect_ai_content.ml_logic.for_images.cnn import load_cnn_model, clean_img_cnn


from detect_ai_content.ml_logic.for_images.vgg16 import Vgg16
from detect_ai_content.ml_logic.for_images.TrueNetImageUsinCustomCNN import TrueNetImageUsinCustomCNN

app = FastAPI()
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

@app.get("/ping")
def ping():
    """
    Preload - models + nltk_data
    """

    # pre-load the light models (not BERT) into memory
    if "TrueNetTextSVC" not in app.state.models:
        app.state.models["TrueNetTextSVC"] = TrueNetTextSVC().local_trained_pipeline()

    if "TrueNetTextLogisticRegression" not in app.state.models:
        app.state.models["TrueNetTextLogisticRegression"] = TrueNetTextLogisticRegression().local_trained_pipeline()

    if "TrueNetTextDecisionTreeClassifier" not in app.state.models:
        app.state.models["TrueNetTextDecisionTreeClassifier"] = TrueNetTextDecisionTreeClassifier().local_trained_pipeline()

    if "TrueNetTextKNeighborsClassifier" not in app.state.models:
        app.state.models["TrueNetTextKNeighborsClassifier"] = TrueNetTextKNeighborsClassifier().local_trained_pipeline()

    if "TrueNetTextRNN" not in app.state.models:
        app.state.models["TrueNetTextRNN"] = TrueNetTextRNN().local_trained_pipeline()

    if "TrueNetTextTfidfNaiveBayesClassifier" not in app.state.models:
        app.state.models["TrueNetTextTfidfNaiveBayesClassifier"] = TrueNetTextTfidfNaiveBayesClassifier().local_trained_pipeline()

    return {}

# http://127.0.0.1:8000/predict?text=lsjisefohlksdjf

@app.get("/text_single_predict")
def predict(
        text: str
    ):
    """
    Make a single prediction prediction (using our best estimator)
    Assumes `text` is provided as a string
    """

    if "TrueNetTextLogisticRegression" not in app.state.models:
        app.state.models["TrueNetTextLogisticRegression"] = TrueNetTextLogisticRegression().local_trained_pipeline()
    best_model = app.state.models["TrueNetTextLogisticRegression"]

    text_df = pd.DataFrame(data=[text],columns=['text'])
    y_pred = best_model.predict(text_df)

    print(f"one pred: {y_pred[0]}")
    return {
        'prediction': int(y_pred[0])
    }

@app.get("/text_multi_predict")
def predict(
        text: str
    ):

    """
    - make a single prediction prediction using user provided 'text'
    - text : text type only  # TO BE IMPROVED for csv data type etc ...

    """

    predictions = {}

    text_df = pd.DataFrame(data=[text],columns=['text'])

    text_enriched_df = enrich_text(text_df)
    # text_enriched_df = enrich_text_BERT_predictions(text_enriched_df)
    text_enriched_df = enrich_lexical_diversity_readability(text_enriched_df)

    if "TrueNetTextSVC" not in app.state.models:
        app.state.models["TrueNetTextSVC"] = TrueNetTextSVC().local_trained_pipeline()
    model = app.state.models["TrueNetTextSVC"]
    y_pred = model.predict(text_enriched_df)
    predictions["TrueNetTextSVC"] = int(y_pred[0])

    if "TrueNetTextLogisticRegression" not in app.state.models:
        app.state.models["TrueNetTextLogisticRegression"] = TrueNetTextLogisticRegression().local_trained_pipeline()
    model = app.state.models["TrueNetTextLogisticRegression"]
    y_pred = model.predict(text_enriched_df)
    predictions["TrueNetTextLogisticRegression"] = int(y_pred[0])

    if "TrueNetTextDecisionTreeClassifier" not in app.state.models:
        app.state.models["TrueNetTextDecisionTreeClassifier"] = TrueNetTextDecisionTreeClassifier().local_trained_pipeline()
    model = app.state.models["TrueNetTextDecisionTreeClassifier"]
    y_pred = model.predict(text_enriched_df)
    predictions["TrueNetTextDecisionTreeClassifier"] = int(y_pred[0])

    if "TrueNetTextKNeighborsClassifier" not in app.state.models:
        app.state.models["TrueNetTextKNeighborsClassifier"] = TrueNetTextKNeighborsClassifier().local_trained_pipeline()
    model = app.state.models["TrueNetTextKNeighborsClassifier"]
    y_pred = model.predict(text_enriched_df)
    predictions["TrueNetTextKNeighborsClassifier"] = int(y_pred[0])

    if "TrueNetTextRNN" not in app.state.models:
        app.state.models["TrueNetTextRNN"] = TrueNetTextRNN().local_trained_pipeline()
        model = app.state.models["TrueNetTextRNN"]
        y_pred = model.predict(text_enriched_df)
        predictions["TrueNetTextRNN"] = int(y_pred[0])

    # Using only "BERT"

    # if "TrueNetTextUsingBERTMaskedPredictions" not in app.state.models:
    #     app.state.models["TrueNetTextUsingBERTMaskedPredictions"] = TrueNetTextUsingBERTMaskedPredictions().local_trained_pipeline()
    # model = app.state.models["TrueNetTextUsingBERTMaskedPredictions"]
    # y_pred = model.predict(text_enriched_df)
    # predictions["TrueNetTextUsingBERTMaskedPredictions"] = int(y_pred[0])

    # Using raw data

    if "TrueNetTextTfidfNaiveBayesClassifier" not in app.state.models:
        app.state.models["TrueNetTextTfidfNaiveBayesClassifier"] = TrueNetTextTfidfNaiveBayesClassifier().local_trained_pipeline()
    model = app.state.models["TrueNetTextTfidfNaiveBayesClassifier"]
    y_pred = model.predict(text_enriched_df)
    predictions["TrueNetTextTfidfNaiveBayesClassifier"] = int(y_pred[0])

    y_preds = []
    number_of_zeros = float(0)
    number_of_ones = float(0)

    for estimator in predictions:
        v = predictions[estimator]
        y_preds.append(v)
        if v == 0:
            number_of_zeros += 1
        else:
            number_of_ones += 1

    prediction = -1
    prediction_confidence = 0

    if np.mean(y_preds) < 0.5:
        prediction = 0
        prediction_confidence = round(100 * (number_of_zeros/len(predictions)))

    else:
        prediction = 1
        prediction_confidence = round(100 * (number_of_ones/len(predictions)))

    print(f"preds: {predictions}")
    return {
        'predictions_details': predictions,
        'prediction': {
            'final_prediction':prediction,
            'final_prediction_confidence':f'{prediction_confidence}%',
        }
    }


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
