
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

from detect_ai_content.ml_logic.for_texts.using_vectorizer_and_NaiveBayes.using_vectorizer_and_NaiveBayes import *

def get_huggingface_texts(number):
    path = "./raw_data/huggingface.co_human_ai_generated_text/model_training_dataset.csv"

    # Extract a small dataset
    chunksize = number
    huggingface_df = None

    for chunk in pd.read_csv(path, chunksize=chunksize):
        # chunk is a DataFrame. To "process" the rows in the chunk:
        huggingface_df = chunk
        break

    huggingface_human_text_df = huggingface_df[["id", "human_text"]]
    huggingface_human_text_df = huggingface_human_text_df.rename(columns={'human_text':'text'},)
    huggingface_human_text_df['generated'] = 0

    huggingface_ai_text_df = huggingface_df[["id", "ai_text"]]
    huggingface_ai_text_df = huggingface_ai_text_df.rename(columns={'ai_text':'text'},)
    huggingface_ai_text_df['generated'] = 1

    huggingface_sample_text_df = pd.concat(objs=[huggingface_human_text_df, huggingface_ai_text_df])
    return huggingface_sample_text_df

class TestTextModelingVectorizer(unittest.TestCase):
    def test_training(self):
        """
        Test the method to Train & validate the model
        """
        df = get_huggingface_texts(10000)
        X = df[['text']]
        y = df['generated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = train_model(X_train, y_train)
        results = evaluate_model(model=model, X_test=X_test, y_test=y_test)
        print(results)

if __name__ == '__main__':
    unittest.main()
