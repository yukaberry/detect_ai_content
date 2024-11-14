import pandas as pd
import numpy as np
import os

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
    predict_probas = best_model.predict_proba(text_df)[0]
    print(f'predict_probas:{predict_probas}')

    predicted_class = int(y_pred[0])
    if predicted_class == 0:
        predict_proba_class = f'{np.round(100 * predict_probas[0])}%'
    else:
        predict_proba_class = f'{np.round(100 * predict_probas[1])}%'

    return {
        'prediction': predicted_class,
        'predict_proba': predict_proba_class,
        'details': {
            'model_name': 'TrueNetTextLogisticRegression',
            'version': 'v0.134'
        }
    }

def prediction_to_result(model, model_name, df):
    y_pred = model.predict(df)
    predicted_class = int(y_pred[0])
    model_prediction_dict = {}
    model_prediction_dict["predicted_class"] = predicted_class

    if model_name == "TrueNetTextRNN":
        print("TrueNetTextRNN special case")
        print(f"y_pred {y_pred}")
        predicted_proba = float(y_pred[0])
        model_prediction_dict["predict_proba_class"] = f'{np.round(100 * predicted_proba)}%'
        if predicted_proba < 0.5:
            model_prediction_dict["predicted_class"] = 0
        else:
            model_prediction_dict["predicted_class"] = 1

    else:
        predict_probas = model.predict_proba(df)[0]
        predicted_class = int(y_pred[0])
        if predicted_class == 0:
            predict_proba_class = f'{np.round(100 * predict_probas[0])}%'
        else:
            predict_proba_class = f'{np.round(100 * predict_probas[1])}%'

        model_prediction_dict["predict_proba_class"] = predict_proba_class

    model_prediction_dict["model_name"] = model_name
    return model_prediction_dict

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
    predictions['TrueNetTextSVC'] = prediction_to_result(model, 'TrueNetTextSVC', text_enriched_df)

    if "TrueNetTextLogisticRegression" not in app.state.models:
        app.state.models["TrueNetTextLogisticRegression"] = TrueNetTextLogisticRegression().local_trained_pipeline()
    model = app.state.models["TrueNetTextLogisticRegression"]
    predictions['TrueNetTextLogisticRegression'] = prediction_to_result(model, 'TrueNetTextLogisticRegression', text_enriched_df)

    if "TrueNetTextDecisionTreeClassifier" not in app.state.models:
        app.state.models["TrueNetTextDecisionTreeClassifier"] = TrueNetTextDecisionTreeClassifier().local_trained_pipeline()
    model = app.state.models["TrueNetTextDecisionTreeClassifier"]
    predictions['TrueNetTextDecisionTreeClassifier'] = prediction_to_result(model, 'TrueNetTextDecisionTreeClassifier', text_enriched_df)

    if "TrueNetTextKNeighborsClassifier" not in app.state.models:
        app.state.models["TrueNetTextKNeighborsClassifier"] = TrueNetTextKNeighborsClassifier().local_trained_pipeline()
    model = app.state.models["TrueNetTextKNeighborsClassifier"]
    predictions['TrueNetTextKNeighborsClassifier'] = prediction_to_result(model, 'TrueNetTextKNeighborsClassifier', text_enriched_df)

    if "TrueNetTextRNN" not in app.state.models:
        app.state.models["TrueNetTextRNN"] = TrueNetTextRNN().local_trained_pipeline()
    model = app.state.models["TrueNetTextRNN"]
    predictions['TrueNetTextRNN'] = prediction_to_result(model, 'TrueNetTextRNN', text_enriched_df)


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
    predictions['TrueNetTextTfidfNaiveBayesClassifier'] = prediction_to_result(model, 'TrueNetTextTfidfNaiveBayesClassifier', text_enriched_df)

    y_preds = []
    number_of_zeros = float(0)
    number_of_ones = float(0)

    for estimator in predictions:
        estimator_result = predictions[estimator]
        v = estimator_result['predicted_class']
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

    # print(f"preds: {predictions}")

    return {
        'prediction': prediction,
        'predict_proba': f'{prediction_confidence}%',
        'details': {
            'models': predictions
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

@app.get("/random_text")
def get_random_text(source: str):
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)
    df = pd.read_csv(f'{module_dir_path}/daigt-v2-samples.csv')
    filtered_df = df[df["source"] == source] # chat_gpt_moth
    # print('random_text filtered_df')
    # print(filtered_df)
    return {
        'text': filtered_df.sample(1).iloc[0]['text']
    }
