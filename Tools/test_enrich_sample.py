
import sys
import os

import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text
from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_masked_words_BERT_prediction

chunksize = 10

def save(df, path):
    if os.path.isfile(path) == False:
        df.to_csv(path, mode='w', index=True, header=True)
    else:
        df.to_csv(path, mode='a', index=True, header=False)

actual_dataset_size = 0

from_filepath = './raw_data/samples/sample_dataset_10000.csv'
to_enrich_filepath = from_filepath.replace('.csv', '_enriched.csv')

for chunk in pd.read_csv(from_filepath, chunksize=chunksize):
    df = chunk

    enriched_df = enrich_text(df)

    number_of_tests = []
    number_of_correct_predictions = []
    pourcentage_of_correct_predictions = []

    for (index, row) in df.iterrows():
        text = row['text']
        (number_of_test, number_of_correct_prediction) = compute_masked_words_BERT_prediction(text)
        pourcentage = 0
        if number_of_test > 0:
            pourcentage = round(100 * number_of_correct_prediction/number_of_test)
        else:
            pourcentage = -1
        number_of_tests.append(number_of_test)
        number_of_correct_predictions.append(number_of_correct_prediction)
        pourcentage_of_correct_predictions.append(pourcentage)

    enriched_df['BERT_number_of_prediction'] = number_of_tests
    enriched_df['BERT_number_of_correct_prediction'] = number_of_correct_predictions
    enriched_df['pourcentage_of_correct_prediction'] = pourcentage_of_correct_predictions

    save(enriched_df, to_enrich_filepath)
    actual_dataset_size += chunksize
    print(f"actual_dataset_size : {actual_dataset_size}")
