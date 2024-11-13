import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text, enrich_lexical_diversity_readability
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer

def smartEnrichFunction(data):
    '''
        Create features if they don't exist
    '''
    data_processed = data.copy()
    if 'repetitions_ratio' not in data_processed:
        data_processed = enrich_text(data_processed)

    if 'lexical_diversity' not in data_processed:
        data_processed = enrich_lexical_diversity_readability(data_processed)

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

def dataframeToSerie(data):
    first_column = data.columns[0]
    return pd.Series(data=data[first_column])

def dataframeToSerieTransformer():
    return FunctionTransformer(dataframeToSerie)
