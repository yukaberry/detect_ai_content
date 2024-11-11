
import sys
import os

import pandas as pd

from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions

from sklearn.model_selection import train_test_split

print(sys.argv)

from_filepath = './raw_data/texts_merged_dataset_enriched.csv'
df = pd.read_csv(from_filepath)

df_train, df_test = train_test_split(df, test_size=0.2)

print(f'df_train.shape: {df_train.shape}')
print(f'df_test.shape: {df_test.shape}')

df_train.to_csv('./raw_data/texts_merged_dataset_enriched/texts_merged_dataset_enriched_train.csv')
df_test.to_csv('./raw_data/texts_merged_dataset_enriched/texts_merged_dataset_enriched_test.csv')
