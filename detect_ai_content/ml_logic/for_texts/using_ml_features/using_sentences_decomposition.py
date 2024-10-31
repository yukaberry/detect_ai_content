
from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

import pandas as pd
import os

def train_LogisticRegression_model(X_train_processed, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_processed, y_train)
    return model

def evaluate_model(model, X_test_processed, y_test):
    y_pred = model.predict(X_test_processed)
    return {
        "accuracy_score": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_score" : f1_score(y_true=y_test, y_pred=y_pred),
        "precision_score": precision_score(y_true=y_test, y_pred=y_pred),
        "recall_score" : recall_score(y_true=y_test, y_pred=y_pred)
    }

def _chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def to_sentences(data, include_generated=True):
    sentences = []
    generated = []

    for index, row in data.iterrows():
        try:
            for s in extract_sentences(row['text']):
                # print(f"START:{s.text}:END")
                # sentences.append(s.text)
                print(f"START:{s}:END")
                if len(s) > 512:
                    # a limit of BERT https://github.com/R1j1t/contextualSpellCheck/issues/64
                    for substring in _chunkstring(s, 512):
                        sentences.append(substring)
                        if include_generated:
                            generated.append(row['generated'])
                else:
                    sentences.append(s)
                    if include_generated:
                        generated.append(row['generated'])

        except RuntimeError as e:
            print(e)
            raise

    if include_generated:
        sentences_df = pd.DataFrame(data={'text':sentences, 'generated':generated})
    else:
        sentences_df = pd.DataFrame(data={'text':sentences})
    return sentences_df

def enrich(data, ):
    sentences_df = data.copy()
    sentences_df['punctuations'] = sentences_df['text'].apply(compute_punctuation_in_text)
    sentences_df['neg_sentiment_polarity'] = sentences_df['text'].apply(compute_neg_sentiment_polarity_in_text)
    sentences_df['pos_sentiment_polarity'] = sentences_df['text'].apply(compute_pos_sentiment_polarity_in_text)
    sentences_df['corrections'] = sentences_df['text'].apply(compute_number_of_text_corrections_using_pyspellchecker)
    sentences_df['text_lenght'] = sentences_df['text'].apply(text_lenght)
    return sentences_df

def preprocess(data, execute_enrich=True):
    sentences_df = data.copy()
    if execute_enrich:
        sentences_df = enrich(data)

    sentences_df = sentences_df[[
        'punctuations',
        'neg_sentiment_polarity',
        'pos_sentiment_polarity',
        'corrections',
        'text_lenght'
    ]]

    scaler = RobustScaler()
    return scaler.fit_transform(sentences_df)

def predict(text):
    sentences_df = to_sentences(pd.DataFrame(data={'text':[text]}), include_generated=False)
    sentences_preprocessed_df = preprocess(sentences_df)
    preds = model.predict(X=sentences_preprocessed_df)
