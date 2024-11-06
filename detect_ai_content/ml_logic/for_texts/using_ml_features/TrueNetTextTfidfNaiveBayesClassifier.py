
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
import os

import mlflow
from mlflow import MlflowClient

from detect_ai_content.params import *
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.evaluation import evaluate_model

class TrueNetTextTfidfNaiveBayesClassifier:
    def _load_model(self):
        """
        Model sumary :
            Trained in 2,532,099 texts (using 3 datasets combined)
            Algo : TfidfVectorizer() + MultinomialNB
            Cross Validate average result (0.2 test) : 0.95
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage="Production")

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextTfidfNaiveBayesClassifier"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextTfidfNaiveBayesClassifier"
        self.mlflow_experiment = "TrueNetTextTfidfNaiveBayesClassifier_experiment_leverdewagon"
        self.model = self._load_model()

    def retrain_full_model():
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        # init
        client = MlflowClient()
        futur_obj = TrueNetTextTfidfNaiveBayesClassifier()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        big_df = get_enriched_df()

        pipeline_naive_bayes = make_pipeline(
            TfidfVectorizer(min_df=0.1),
            MultinomialNB()
        )

        X = big_df['text']
        y = big_df['generated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = pipeline_naive_bayes.fit(X=X_train, y=y_train)

        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))

        # mlflow_save_params
        mlflow_save_params(
            training_set_size= X_test.shape[0],
            row_count= big_df.shape[0],
            dataset_huggingface_human_ai_generated_text=True,
            dataset_kaggle_ai_generated_vs_human_text=True,
            dataset_kaggle_daigt_v2_train_dataset=True,
            additional_parameters={
                'preprocess':'nothing - no cleaning'
            }
        )

        results = evaluate_model(model, X_test, y_test)

        # mlflow_save_metrics
        mlflow_save_metrics(f1_score= results['f1_score'],
                            recall_score= results['recall_score'],
                            precision_score= results['precision_score'],
                            accuracy_score= results['accuracy_score'])


        # mlflow_save_model
        input_example_df = big_df.sample(3)

        mlflow_save_model(
            model=model,
            is_tensorflow=False,
            model_name=futur_obj.mlflow_model_name,
            input_example=input_example_df
        )

        mlflow.end_run()
