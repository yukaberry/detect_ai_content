
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pandas as pd
import numpy as np
import pickle
import os

def train_model(X, y):
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    model = pipeline_naive_bayes.fit(X=X['text'].values, y=y.values)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test['text'].values)

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

    df = pd.read_csv(f'{module_dir_path}/../raw_data/huggingface.co_human_ai_generated_text/model_training_dataset_enriched.csv')
    AI_Human_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/kaggle-ai-generated-vs-human-text/AI_Human_enriched.csv')
    daigt_v2_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/daigt-v2-train-dataset/train_v2_drcat_02_enriched.csv')
    big_df = pd.concat(objs=[df, AI_Human_enriched_df, daigt_v2_enriched_df])

    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    X = big_df['text']
    y = big_df['generated']

    model = pipeline_naive_bayes.fit(X=X, y=y)

    model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_TfidfVectorizer_MultinomialNB.pickle'
    pickle.dump(model, open(model_path, 'wb'))
