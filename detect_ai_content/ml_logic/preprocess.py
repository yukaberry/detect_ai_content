import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer

def preprocess(data, auto_enrich=True):
    data_processed = data.copy()
    if auto_enrich:
        data_processed = enrich_text(data_processed)
        data_processed = enrich_text_BERT_predictions(data_processed)

    # keep only the features we want
    data_processed = data_processed[[
        'repetitions_ratio',
        'punctuations_ratio',
        'text_corrections_ratio',
        'average_sentence_lenght',
        'average_neg_sentiment_polarity',
        'pourcentage_of_correct_prediction'
    ]]

    scaler = RobustScaler()
    return scaler.fit_transform(data_processed)

def smartEnrichFunction(data):
    '''
        Create features if they don't exist
    '''
    data_processed = data.copy()
    if 'repetitions_ratio' not in data_processed:
        data_processed = enrich_text(data_processed)
    if 'pourcentage_of_correct_prediction' not in data_processed:
        data_processed = enrich_text_BERT_predictions(data_processed)

    return data_processed

def smartEnrichTransformer():
    return FunctionTransformer(smartEnrichFunction)

def smartCleanerFunction(data):
    '''
        Create features if they don't exist
    '''
    text_df = data['text']
    data_cleaned = data[text_df.duplicated() == False]
    return data_cleaned

def smartCleanerTransformer():
    return FunctionTransformer(smartCleanerFunction)

def smartSelectionFunction(data, columns):
    '''
        Create features if they don't exist
    '''

    cleaned_data = data.copy()
    cleaned_data = cleaned_data[columns]
    return cleaned_data

def smartSelectionTransformer(columns: None):
    return FunctionTransformer(smartSelectionFunction, kw_args={'columns':columns})
