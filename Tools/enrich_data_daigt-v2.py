
import sys
import os

import pandas as pd

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import enrich

print(sys.argv)

from_filepath = sys.argv[1]
to_enrich_filepath = from_filepath.replace('.csv', '_enriched.csv')

chunksize = 10

def save(df, path):
    if os.path.isfile(path) == False:
        df.to_csv(path, mode='w', index=True, header=True)
    else:
        df.to_csv(path, mode='a', index=True, header=False)

actual_dataset_size = 0

for chunk in pd.read_csv(from_filepath, chunksize=chunksize):
    df = chunk[["text"]]
    df['generated'] = 1
    df = enrich(df)

    save(df, to_enrich_filepath)
    actual_dataset_size += chunksize
    print(f"actual_dataset_size : {actual_dataset_size}")
