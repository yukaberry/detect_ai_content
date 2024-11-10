

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_masked_words_BERT_prediction
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.preprocess import preprocess
from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions

import pandas as pd
import os
import mlflow
from mlflow import MlflowClient
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Normalization
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Flatten, Input
from keras.callbacks import EarlyStopping

class TrueNetTextRNN:
    def _local_model(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        local_path = f'{module_dir_path}/models/leverdewagon/{self.mlflow_model_name}.pickle'
        latest_model = pickle.load(open(local_path, 'rb'))
        return latest_model

    def _load_model(self, stage="Production"):
        """
        Model sumary :
            Trained
            Algo :
            Cross Validate average result (0.2 test) :
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def get_architecture_model():
        model = Sequential()
        model.add(Normalization(input_shape = (6, )))
        model.add(Dense(units=12, activation="relu"))
        model.add(Dense(units=24, activation="relu"))
        model.add(Dense(units=12, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        return model

    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextRNN"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextRNN"
        self.mlflow_experiment = "TrueNetTextRNN_experiment_leverdewagon"
        self.model = self._load_model()

    def preprocess(data):
        scaler = RobustScaler()
        return scaler.fit_transform(
            data[TrueNetTextRNN.columns()])

    def columns():
        return [
                'repetitions_ratio',
                'punctuations_ratio',
                'text_corrections_ratio',
                'average_sentence_lenght',
                'average_neg_sentiment_polarity',
                'pourcentage_of_correct_prediction'
            ]

    def retrain_full_model():
        print("retrain_full_model START")
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        futur_obj = TrueNetTextRNN()

        df = get_enriched_df(purpose="train")
        y = df['generated']
        X = df[TrueNetTextRNN.columns()] # TrueNetTextRNN.preprocess(data=df)

        model = TrueNetTextRNN.get_architecture_model()
        model = TrueNetTextRNN.compile_model(model)
        print(model.summary())

        es = EarlyStopping(monitor="val_loss", patience=20)
        print(X.head())
        print(y.head())

        history = model.fit(x=X,
                            y=y,
                            epochs=1000,
                            validation_split=0.2,
                            callbacks=[es],
                            verbose=1)

        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))
