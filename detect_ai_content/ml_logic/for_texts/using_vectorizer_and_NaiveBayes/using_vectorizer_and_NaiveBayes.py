
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
import os

import mlflow
from mlflow import MlflowClient

from detect_ai_content.params import *
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params

def train_model(X, y):
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    model = pipeline_naive_bayes.fit(X=X['text'].values, y=y.values)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy_score": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_score" : f1_score(y_true=y_test, y_pred=y_pred),
        "precision_score": precision_score(y_true=y_test, y_pred=y_pred),
        "recall_score" : recall_score(y_true=y_test, y_pred=y_pred)
    }

def load_model():
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)
    model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_TfidfVectorizer_MultinomialNB.pickle'
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

def retrain_full_model():
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)

    # init
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(MLFLOW_VECTORIZER_EXPERIMENT).experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    # Load huggingface dataset
    print("Load huggingface.co_human_ai_generated_text")
    huggingface_df = pd.read_csv(f'{module_dir_path}/../raw_data/huggingface.co_human_ai_generated_text/model_training_dataset.csv')
    huggingface_human_text_df = huggingface_df[["human_text"]]
    huggingface_human_text_df = huggingface_human_text_df.rename(columns={'human_text':'text'},)
    huggingface_human_text_df['generated'] = 0
    huggingface_ai_text_df = huggingface_df[["ai_text"]]
    huggingface_ai_text_df = huggingface_ai_text_df.rename(columns={'ai_text':'text'},)
    huggingface_ai_text_df['generated'] = 1

    # Load kaggle dataset
    print("Load kaggle-ai-generated-vs-human-text")
    AI_Human_df = pd.read_csv(f'{module_dir_path}/../raw_data/kaggle-ai-generated-vs-human-text/AI_Human.csv')
    AI_Human_df = AI_Human_df[["text", "generated"]]

    # Load kaggle dataset
    print("daigt-v2-train-dataset")
    daigt_v2_df = pd.read_csv(f'{module_dir_path}/../raw_data/daigt-v2-train-dataset/train_v2_drcat_02.csv')
    daigt_v2_df = daigt_v2_df[["text"]]
    daigt_v2_df['generated'] = 1

    big_df = pd.concat(objs=[
        huggingface_human_text_df,
        huggingface_ai_text_df,
        AI_Human_df,
        daigt_v2_df
        ])

    # big_df = big_df.sample(500_000)

    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(min_df=0.1),
        MultinomialNB()
    )

    X = big_df['text']
    y = big_df['generated']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = pipeline_naive_bayes.fit(X=X_train, y=y_train)

    model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_TfidfVectorizer_MultinomialNB.pickle'
    pickle.dump(model, open(model_path, 'wb'))

    # mlflow_save_params
    mlflow_save_params(
        training_set_size= X_test.shape[0],
        row_count= big_df.shape[0],
        dataset_huggingface_human_ai_generated_text=True,
        dataset_kaggle_ai_generated_vs_human_text=True,
        dataset_kaggle_daigt_v2_train_dataset=True,
        additional_parameters={
            'preprocess':'nothing - no cleaning'
        }
    )

    results = evaluate_model(model, X_test, y_test)

    # mlflow_save_metrics
    mlflow_save_metrics(f1_score= results['f1_score'],
                        recall_score= results['recall_score'],
                        precision_score= results['precision_score'],
                        accuracy_score= results['accuracy_score'])


    # mlflow_save_model
    input_example_df = big_df.sample(3)

    mlflow_save_model(
        model=model,
        is_tensorflow=False,
        model_name=MLFLOW_VECTORIZER_MODEL_NAME,
        input_example=input_example_df
    )

    mlflow.end_run()
