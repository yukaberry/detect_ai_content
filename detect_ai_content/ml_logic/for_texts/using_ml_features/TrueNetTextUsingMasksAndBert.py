

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_masked_words_BERT_prediction
from detect_ai_content.ml_logic.data import get_enriched_df

import pandas as pd
import os

class TrueNetTextUsingMasksAndBert:
    def _load_model(self):
        """
        Model sumary :
            Trained TBD
            Algo : TBD
            Cross Validate average result (0.2 test) : TBD
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage="Production")

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextUsingMasksAndBert"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextUsingMasksAndBert"
        self.mlflow_experiment = "TrueNetTextUsingMasksAndBert_experiment_leverdewagon"
        self.model = self._load_model()

    # inspiration : https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
    def run_one_test():
        # text = "Egypt is a land of wonders and fascinations, with centuries of history brought to life in its pyramids, museums, and cultural sites. And a visit to the markets and souks of Generic_City are a must-see when visiting this historic city. Beaches are plentiful in Egypt, from the sleek white sand of Sinai on the Red Sea to the breathtaking turquoise of the Mediterranean coast near Alexdandria. Whether looking for a taste of modern luxury or a rustic and more traditional experience, the best beach locations in Egypt will have something for everyone."
        # compute_masked_words_BERT_prediction(text)
        print("START loading ")
        # df = get_enriched_df(100)
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        df = pd.read_csv(f'{module_dir_path}/../raw_data/samples/sample_dataset_1000.csv')
        df = df.sample(100)
        print("END loading")

        items = []
        iteration = 0
        for (index, row) in df.iterrows():
            iteration += 1
            item = {}
            text = row['text']
            generated = row['generated']
            item["text"] = text
            item["generated"] = generated

            print(f"🔎 index: {iteration}")
            (number_of_test, number_of_correct_prediction) = compute_masked_words_BERT_prediction(text)
            pourcentage = round(100 * number_of_correct_prediction/number_of_test)
            print(f'{generated}\t{pourcentage}')

            item["number_of_test"] = number_of_test
            item["number_of_correct_prediction"] = number_of_correct_prediction
            item["pourcentage_of_correct_prediction"] = pourcentage
            items.append(item)

        output_df = pd.DataFrame(data=items)
        to_path = f'{module_dir_path}/../raw_data/bert_prediction_tests.csv'
        output_df.to_csv(to_path)
