
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
import os

import mlflow
from mlflow import MlflowClient

from detect_ai_content.params import *
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.evaluation import evaluate_model
from detect_ai_content.ml_logic.preprocess import smartSelectionTransformer, dataframeToSerieTransformer

from sklearn.pipeline import make_pipeline, Pipeline

class TrueNetTextTfidfNaiveBayesClassifier:
    def local_trained_pipeline(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        self.model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{self.mlflow_model_name}_pipeline.pickle'
        return pickle.load(open(self.model_path, 'rb'))

    def st_size(self):
        import os
        return os.stat(self.model_path).st_size

    def get_mlflow_model(self, stage="Production"):
        """
        Model sumary :
            Trained in 2,532,099 texts (using 3 datasets combined)
            Algo : TfidfVectorizer() + MultinomialNB
            Cross Validate average result (0.2 test) : 0.95
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextTfidfNaiveBayesClassifier"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextTfidfNaiveBayesClassifier"
        self.mlflow_experiment = "TrueNetTextTfidfNaiveBayesClassifier_experiment_leverdewagon"
        self.model = self.local_trained_pipeline()

    def retrain_full_model():
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        # init
        client = MlflowClient()
        futur_obj = TrueNetTextTfidfNaiveBayesClassifier()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        big_df = get_enriched_df(purpose="train")

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
            training_fit_size = X_train.shape[0],
            training_test_size = X_test.shape[0],
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


    def retrain_production_pipeline():
        columns = [
            'text'
        ]

        features_selection_transformer = smartSelectionTransformer(columns=columns)

        pipeline = Pipeline([
            ('features_selection', features_selection_transformer),
            ('dataframeToSerieTransformer', dataframeToSerieTransformer()),
            ('TfidfVectorizer', TfidfVectorizer(min_df=0.1)),
            ('MultinomialNB', MultinomialNB())
        ])

        df = get_enriched_df(purpose="train")
        y = df['generated']
        X = df

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X=X_train, y=y_train)

        results = evaluate_model(pipeline, X_test, y_test)
        print(results)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        mlflow_model_name = TrueNetTextTfidfNaiveBayesClassifier().mlflow_model_name
        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{mlflow_model_name}_pipeline.pickle'
        pickle.dump(pipeline, open(model_path, 'wb'))
