
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.params import *
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model

import os
import pickle
import mlflow
from mlflow import MlflowClient

class TrueNetTextSVC:
    def _load_model(self):
        """
        Model sumary :
            Trained TBD
            Algo : TBD
            Cross Validate average result (0.2 test) : TBD
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage="Production")

    def __init__(self):
        self.name = "TrueNetTextSVC"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextSVC"
        self.mlflow_experiment = "TrueNetTextSVC_experiment_leverdewagon"
        self.model = self._load_model()

    def run_grid_search():
        param_grid = {
            'C':[1,10,100,1000],
            'gamma':[1,0.1,0.001,0.0001],
            'kernel':['linear','rbf']
        }

        df = get_enriched_df()
        X = df[[
            'repetitions_ratio',
            'punctuations_ratio',
            'text_corrections_ratio',
            'average_sentence_lenght',
            'average_neg_sentiment_polarity',
        ]]
        y = df['generated']
        grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
        grid.fit(X,y)
        print(grid.best_estimator_)

    def retrain_full_model():

        # init
        client = MlflowClient()
        futur_obj = TrueNetTextSVC()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)


        big_df = get_enriched_df(10_000)

        pipeline = make_pipeline(
            RobustScaler(),
            SVC(C=10, gamma=0.1) # param from run_grid_search
        )

        X = big_df[[
            'repetitions_ratio',
            'punctuations_ratio',
            'text_corrections_ratio',
            'average_sentence_lenght',
            'average_neg_sentiment_polarity',
        ]]
        y = big_df['generated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = pipeline.fit(X_train, y_train)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))

        # mlflow_save_params
        mlflow_save_params(
            training_set_size= X_test.shape[0],
            row_count= big_df.shape[0],
            dataset_huggingface_human_ai_generated_text=True,
            dataset_kaggle_ai_generated_vs_human_text=True,
            dataset_kaggle_daigt_v2_train_dataset=True,
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