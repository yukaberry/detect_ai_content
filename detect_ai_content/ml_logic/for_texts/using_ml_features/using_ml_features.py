
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
from mlflow import MlflowClient

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params
from detect_ai_content.params import *

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

def enrich(data):
    data_enriched = data.copy()
    # print('enrich compute_punctuation_in_text')
    data_enriched['punctuations_nb'] = data_enriched['text'].apply(compute_punctuation_in_text)

    # print('enrich compute_neg_sentiment_polarity_in_text')
    data_enriched['neg_sentiment_polarity'] = data_enriched['text'].apply(compute_neg_sentiment_polarity_in_text)

    # print('enrich compute_pos_sentiment_polarity_in_text')
    data_enriched['pos_sentiment_polarity'] = data_enriched['text'].apply(compute_pos_sentiment_polarity_in_text)

    # print('enrich text_corrections')
    data_enriched['text_corrections_nb'] = data_enriched['text'].apply(compute_number_of_text_corrections_using_nltk_words)

    # print('enrich compute_repetitions_in_text')
    data_enriched['text_repetitions_nb'] = data_enriched['text'].apply(compute_repetitions_in_text)

    # print('enrich number_of_sentences')
    data_enriched['number_of_sentences'] = data_enriched['text'].apply(number_of_sentences)

    # print('enrich text_lenght')
    data_enriched['text_lenght'] = data_enriched['text'].apply(text_lenght)

    return data_enriched

def load_model():
    """
        Model sumary :
            Trained in 2,532,099 texts (using 3 datasets combined)
            Algo : TfidfVectorizer() + MultinomialNB
            Cross Validate average result (0.2 test) : 0.83
    """
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)
    model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_ml_features.pickle'
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

def preprocess(data, auto_enrich=True):
    if auto_enrich:
        data_processed = enrich(data=data)
    else:
        data_processed = data

    data_processed['repetitions_ratio'] = data_processed['text_repetitions_nb']/data_processed['text_lenght']
    data_processed['punctuations_ratio'] = data_processed['punctuations_nb']/data_processed['text_lenght']
    data_processed['text_corrections_ratio'] = data_processed['text_corrections_nb']/data_processed['text_lenght']
    data_processed['average_sentence_lenght'] = data_processed['text_lenght']/data_processed['number_of_sentences']
    data_processed['average_neg_sentiment_polarity'] = data_processed['neg_sentiment_polarity']/data_processed['text_lenght']

    data_processed = data_processed[[
        'repetitions_ratio',
        'punctuations_ratio',
        'text_corrections_ratio',
        'average_sentence_lenght',
        'average_neg_sentiment_polarity'
    ]]

    scaler = RobustScaler()
    return scaler.fit_transform(data_processed)

def retrain_full_model():
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)

    # init
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment_id = client.get_experiment_by_name(MLFLOW_FE_EXPERIMENT).experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    df = pd.read_csv(f'{module_dir_path}/../raw_data/huggingface.co_human_ai_generated_text/model_training_dataset_enriched.csv')

    df['repetitions_ratio'] = df['text_repetitions_nb']/df['text_lenght']
    df['punctuations_ratio'] = df['punctuations_nb']/df['text_lenght']
    df['text_corrections_ratio'] = df['text_corrections_nb']/df['text_lenght']
    df['text_corrections_set_ratio'] = df['text_corrections_nb']/df['number_of_sentences']
    df['average_neg_sentiment_polarity'] = df['neg_sentiment_polarity']/df['text_lenght']

    AI_Human_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/kaggle-ai-generated-vs-human-text/AI_Human_enriched.csv')
    AI_Human_enriched_df['repetitions_ratio'] = AI_Human_enriched_df['text_repetitions_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['punctuations_ratio'] = AI_Human_enriched_df['punctuations_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['text_corrections_ratio'] = AI_Human_enriched_df['text_corrections_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['text_corrections_set_ratio'] = AI_Human_enriched_df['text_corrections_nb']/AI_Human_enriched_df['number_of_sentences']
    AI_Human_enriched_df['average_sentence_lenght'] = AI_Human_enriched_df['text_lenght']/AI_Human_enriched_df['number_of_sentences']
    AI_Human_enriched_df['average_neg_sentiment_polarity'] = AI_Human_enriched_df['neg_sentiment_polarity']/AI_Human_enriched_df['text_lenght']

    daigt_v2_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/daigt-v2-train-dataset/train_v2_drcat_02_enriched.csv')
    daigt_v2_enriched_df['repetitions_ratio'] = daigt_v2_enriched_df['text_repetitions_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['punctuations_ratio'] = daigt_v2_enriched_df['punctuations_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['text_corrections_ratio'] = daigt_v2_enriched_df['text_corrections_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['text_corrections_set_ratio'] = daigt_v2_enriched_df['text_corrections_nb']/daigt_v2_enriched_df['number_of_sentences']
    daigt_v2_enriched_df['average_neg_sentiment_polarity'] = daigt_v2_enriched_df['neg_sentiment_polarity']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['repetitions_ratio'].mean()

    big_df = pd.concat(objs=[df, AI_Human_enriched_df, daigt_v2_enriched_df])
    big_df = big_df[np.isinf(big_df['average_sentence_lenght']) == False]
    big_df = big_df[np.isinf(big_df['repetitions_ratio']) == False]
    big_df = big_df[np.isinf(big_df['punctuations_ratio']) == False]
    big_df = big_df[np.isinf(big_df['text_corrections_ratio']) == False]
    big_df = big_df[np.isinf(big_df['average_neg_sentiment_polarity']) == False]
    big_df = big_df.fillna(0)

    pipeline_linear_regression = make_pipeline(
        RobustScaler(),
        LogisticRegression()
    )

    X = big_df[[
    'repetitions_ratio',
    'punctuations_ratio',
    'text_corrections_ratio',
    'average_sentence_lenght',
    'average_neg_sentiment_polarity',
    ]]
    y = big_df['generated']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = pipeline_linear_regression.fit(X=X_train, y=y_train)

    model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/genai_text_detection_using_ml_features.pickle'
    pickle.dump(model, open(model_path, 'wb'))

    # mlflow_save_params
    mlflow_save_params(
        training_set_size= X_test.shape[0],
        row_count= big_df.shape[0],
        dataset_huggingface_human_ai_generated_text=True,
        dataset_kaggle_ai_generated_vs_human_text=True,
        dataset_kaggle_daigt_v2_train_dataset=True,
    )

    results = evaluate_model(model, X_test, y_test)

    # mlflow_save_metrics
    mlflow_save_metrics(f1_score= results['f1_score'],
                        recall_score= results['recall_score'],
                        precision_score= results['precision_score'],
                        accuracy_score= results['accuracy_score'])

    # mlflow_save_model
    example_df = big_df.sample(3)

    mlflow_save_model(
        model=model,
        is_tensorflow=False,
        model_name=MLFLOW_FE_MODEL_NAME,
        input_example=example_df
    )

    mlflow.end_run()
