
import sys
import os

import pandas as pd

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import enrich

print(sys.argv)

from_filepath = sys.argv[1]
to_enrich_filepath = from_filepath.replace('.csv', '_enriched_v2.csv')

chunksize = 10

def save(df, path):
    if os.path.isfile(path) == False:
        df.to_csv(path, mode='w', index=True, header=True)
    else:
        df.to_csv(path, mode='a', index=True, header=False)

actual_dataset_size = 0

for chunk in pd.read_csv(from_filepath, chunksize=chunksize):
    huggingface_df = chunk

    huggingface_human_text_df = huggingface_df[["id", "human_text"]]
    huggingface_human_text_df = huggingface_human_text_df.rename(columns={'human_text':'text'},)
    huggingface_human_text_df['generated'] = 0

    huggingface_ai_text_df = huggingface_df[["id", "ai_text"]]
    huggingface_ai_text_df = huggingface_ai_text_df.rename(columns={'ai_text':'text'},)
    huggingface_ai_text_df['generated'] = 1

    huggingface_sample_text_df = pd.concat(objs=[huggingface_human_text_df, huggingface_ai_text_df])
    huggingface_sample_text_df = enrich(huggingface_sample_text_df)

    save(huggingface_sample_text_df, to_enrich_filepath)
    actual_dataset_size += chunksize
    print(f"actual_dataset_size : {actual_dataset_size}")
