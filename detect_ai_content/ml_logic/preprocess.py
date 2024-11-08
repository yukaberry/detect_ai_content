import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions
from sklearn.preprocessing import RobustScaler

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
       # 'average_pos_sentiment_polarity'
    ]]

    scaler = RobustScaler()
    return scaler.fit_transform(data_processed)
