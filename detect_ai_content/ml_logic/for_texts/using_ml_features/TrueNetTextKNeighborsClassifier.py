
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.preprocess import smartCleanerTransformer, smartEnrichTransformer, smartSelectionTransformer, smartEnrichFunction

import os
import pandas as pd
import numpy as np
import pickle
import mlflow
from mlflow import MlflowClient

class TrueNetTextKNeighborsClassifier:
    def local_trained_pipeline(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        self.model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{self.mlflow_model_name}_pipeline.pickle'
        return pickle.load(open(self.model_path, 'rb'))

    def get_mlflow_model(self, stage="Production"):
        """
        Model sumary :
            Trained TBD
            Algo : TBD
            Cross Validate average result (0.2 test) : TBD
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def st_size(self):
        import os
        return os.stat(self.model_path).st_size

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextKNeighborsClassifier"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextKNeighborsClassifier"
        self.mlflow_experiment = "TrueNetTextKNeighborsClassifier_experiment_leverdewagon"
        self.pipeline = self.local_trained_pipeline()

    def run_grid_search():
        df = get_enriched_df()
        X = smartEnrichFunction(data=df, auto_enrich=False)
        y = df['generated']

        model = KNeighborsClassifier()
        k_range = list(range(1, 31))
        param_grid = dict(n_neighbors=k_range)

        grid = GridSearchCV(model ,param_grid,refit = True, verbose=2)
        grid.fit(X,y)
        print(grid.best_estimator_)

    def retrain_full_model():
        # init
        client = MlflowClient()
        futur_obj = TrueNetTextKNeighborsClassifier()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        big_df = get_enriched_df()
        y = big_df['generated']
        X = smartEnrichFunction(data=big_df)

        param_n_neighbors = 20 # param from run_grid_search

        model = KNeighborsClassifier(n_neighbors=param_n_neighbors)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = model.fit(X_train, y_train)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))

        # mlflow_save_params
        additional_parameters = {}
        additional_parameters['model_param_n_neighbors'] = param_n_neighbors

        mlflow_save_params(
            training_fit_size=X_train.shape[0],
            training_test_size=X_test.shape[0],
            row_count= big_df.shape[0],
            dataset_huggingface_human_ai_generated_text=True,
            dataset_kaggle_ai_generated_vs_human_text=True,
            dataset_kaggle_daigt_v2_train_dataset=True,
            additional_parameters=additional_parameters
        )

        results = evaluate_model(model, X_test, y_test)

        # mlflow_save_metrics
        mlflow_save_metrics(f1_score= results['f1_score'],
                            recall_score= results['recall_score'],
                            precision_score= results['precision_score'],
                            accuracy_score= results['accuracy_score'])

        # mlflow_save_model
        example_df = big_df.sample(3)

        mlflow_save_model(
            model=model,
            is_tensorflow=False,
            model_name=futur_obj.mlflow_model_name,
            input_example=example_df
        )

        mlflow.end_run()


    def retrain_production_pipeline():
        columns = [
            'repetitions_ratio',
            'punctuations_ratio',
            'text_corrections_ratio',
            'average_sentence_lenght',
            'average_neg_sentiment_polarity',
            'lexical_diversity',
            'smog_index',
            'flesch_reading_ease',
            'avg_word_length'
        ]

        param_n_neighbors = 20 # param from run_grid_search
        model = KNeighborsClassifier(n_neighbors=param_n_neighbors)

        features_selection_transformer = smartSelectionTransformer(columns=columns)
        pipeline = Pipeline([
            ('row_cleaner', smartCleanerTransformer()),
            ('enricher', smartEnrichTransformer()),
            ('features_selection', features_selection_transformer),
            ('scaler', RobustScaler()),
            ('estimator', model),
             ])

        df = get_enriched_df()
        y = df['generated']

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, )
        pipeline.fit(X=X_train, y=y_train)

        results = evaluate_model(pipeline, X_test, y_test)
        print(results)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        mlflow_model_name = TrueNetTextKNeighborsClassifier().mlflow_model_name
        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{mlflow_model_name}_pipeline.pickle'
        pickle.dump(pipeline, open(model_path, 'wb'))
