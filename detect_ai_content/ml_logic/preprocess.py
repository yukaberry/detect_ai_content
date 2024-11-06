import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text
from sklearn.preprocessing import RobustScaler

def preprocess(data, auto_enrich=True):
    if auto_enrich:
        data_processed = enrich_text(data=data.copy())
    else:
        data_processed = data.copy()

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
