

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import compute_masked_words_BERT_prediction
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.preprocess import preprocess
from detect_ai_content.ml_logic.data import enrich_text, enrich_text_BERT_predictions

import pandas as pd
import os
import mlflow
from mlflow import MlflowClient
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

class TrueNetTextUsingBERTMaskedPredictions:
    def _local_model(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        local_path = f'{module_dir_path}/models/leverdewagon/{self.mlflow_model_name}.pickle'
        latest_model = pickle.load(open(local_path, 'rb'))
        return latest_model

    def _load_model(self, stage="Production"):
        """
        Model sumary :
            Trained 5720 rows
            Algo : BERT predictions + LogisticRegression
            Cross Validate average result (0.2 test) : 0.75
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextUsingBERTMaskedPredictions"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextUsingBERTMaskedPredictions"
        self.mlflow_experiment = "TrueNetTextUsingBERTMaskedPredictions_experiment_leverdewagon"
        self.model = self._load_model()

    # inspiration : https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
    # ../../raw_data/samples/sample_dataset_10000_enriched.csv

    def preprocess(data):
        scaler = RobustScaler()
        return scaler.fit_transform(data[['pourcentage_of_correct_prediction']]) # predict only with BERT predictions

    def retrain_full_model():
        print("retrain_full_model START")
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        df = get_enriched_df()
        df = df[['generated', 'pourcentage_of_correct_prediction']]
        y = df['generated']
        X = TrueNetTextUsingBERTMaskedPredictions.preprocess(data=df)

        # init
        client = MlflowClient()
        futur_obj = TrueNetTextUsingBERTMaskedPredictions()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        model = LogisticRegression()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = model.fit(X=X_train, y=y_train)

        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))

        additional_parameters = {
            'dataset': 'get_enriched_df',
            'using_BERT_masking_predictions': 'pourcentage_of_correct_prediction',
        }

        # mlflow_save_params
        mlflow_save_params(
            training_fit_size=X_train.shape[0],
            training_test_size=X_test.shape[0],
            row_count= df.shape[0],
            dataset_huggingface_human_ai_generated_text=False,
            dataset_kaggle_ai_generated_vs_human_text=False,
            dataset_kaggle_daigt_v2_train_dataset=False,
            additional_parameters=additional_parameters
        )

        results = evaluate_model(model, X_test, y_test)
        print(results)

        # mlflow_save_metrics
        mlflow_save_metrics(f1_score= results['f1_score'],
                            recall_score= results['recall_score'],
                            precision_score= results['precision_score'],
                            accuracy_score= results['accuracy_score'])

        # mlflow_save_model
        example_df = df.sample(3)

        mlflow_save_model(
            model=model,
            is_tensorflow=False,
            model_name=futur_obj.mlflow_model_name,
            input_example=example_df
        )

        mlflow.end_run()
