

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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

class TrueNetTextUsingBERTMaskedPredictions:
    def _local_model(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        local_path = f'{module_dir_path}/models/leverdewagon/{self.mlflow_model_name}.pickle'
        latest_model = pickle.load(open(local_path, 'rb'))
        return latest_model

    def _load_model(self):
        """
        Model sumary :
            Trained 5720 rows
            Algo : BERT predictions + LogisticRegression
            Cross Validate average result (0.2 test) : 0.75
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage="Production")

    def __init__(self):
        self.description = ""
        self.name = "TrueNetTextUsingBERTMaskedPredictions"
        self.description = ""
        self.mlflow_model_name = "TrueNetTextUsingBERTMaskedPredictions"
        self.mlflow_experiment = "TrueNetTextUsingBERTMaskedPredictions_experiment_leverdewagon"
        self.model = self._load_model()

    # inspiration : https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
    # ../../raw_data/samples/sample_dataset_10000_enriched.csv

    def preprocess_data():
        print("preprocess START")
        text = """

In conclusion, adding an extra one and a half hours of school could have serious detrimental effects on the lives of students, both academically and socially.

It is essential for students to be able to use their scarce free time to pursue interests, goals, and relationships, that allow them to have life experiences and develop a better understanding of the world.
Taking an extra one and a half hours per school day would have many pitfalls that students and parents must consider. The lack of available time for leisure activities, relationships, and sports could result in disinterest in school and a reduced capacity to learn. Furthermore, students may not have enough time and energy to participate in sports teams or practice instruments, which can lead to disinterest in their academic career. The lack of available time after school significantly reduces these important opportunities. On top of that, students would be robbed of additional time spent with friends, family, and exploring personal hobbies or leisure activities",1.0,88,53,60
1,"more than core classes cause core classses are required for gradutaion and the elecitves are for fun and i think that the school should give these classes as an elecitve because it will be fun to learn new things and explore what you would be good at.

some people may believe that you should take a music class,or art class,or drama class however; i believe That those classes should be electives cause their less important then the core classes. do you think that these classes should be electives or a core class?.I say that it should be an elecitve because some peole are taking it as electives and so they cahn graduate on time thats why i say they should make theses classes go for elecitives and kids will like more drama and art and or music classes. and you get to meet new and differnt people and thats why it should be an elective.. but at the same time it would be fun to have one of those classes. and so you expiernce something new",0.0,43,21,49

In conclusion, adding an extra one and a half hours of school could have serious detrimental effects on the lives of students, both academically and socially.

It is essential for students to be able to use their scarce free time to pursue interests, goals, and relationships, that allow them to have life experiences and develop a better understanding of the world.
Taking an extra one and a half hours per school day would have many pitfalls that students and parents must consider. The lack of available time for leisure activities, relationships, and sports could result in disinterest in school and a reduced capacity to learn. Furthermore, students may not have enough time and energy to participate in sports teams or practice instruments, which can lead to disinterest in their academic career. The lack of available time after school significantly reduces these important opportunities. On top of that, students would be robbed of additional time spent with friends, family, and exploring personal hobbies or leisure activities",1.0,88,53,60
1,"more than core classes cause core classses are required for gradutaion and the elecitves are for fun and i think that the school should give these classes as an elecitve because it will be fun to learn new things and explore what you would be good at.

some people may believe that you should take a music class,or art class,or drama class however; i believe That those classes should be electives cause their less important then the core classes. do you think that these classes should be electives or a core class?.I say that it should be an elecitve because some peole are taking it as electives and so they cahn graduate on time thats why i say they should make theses classes go for elecitives and kids will like more drama and art and or music classes. and you get to meet new and differnt people and thats why it should be an elective.. but at the same time it would be fun to have one of those classes. and so you expiernce something new",0.0,43,21,49

        """

        df = pd.DataFrame(data=[text], columns=['text'])
        print("enrich_text START")
        enriched_df = enrich_text(data=df)
        print("enrich_text_BERT_predictions START")
        enriched_df = enrich_text_BERT_predictions(data=enriched_df)
        print(enriched_df)
        print("preprocess START")
        print(preprocess(data=enriched_df))

    def retrain_full_model():
        print("retrain_full_model START")
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        # init
        client = MlflowClient()
        futur_obj = TrueNetTextUsingBERTMaskedPredictions()
        experiment_id = client.get_experiment_by_name(futur_obj.mlflow_experiment).experiment_id
        mlflow.start_run(experiment_id=experiment_id)

        enriched_path = f'{module_dir_path}/../raw_data/samples/sample_dataset_10000_enriched.csv'
        big_df = pd.read_csv(enriched_path)

        X = big_df[[
            'pourcentage_of_correct_prediction'
        ]]

        y = big_df['generated']

        pipeline_linear_regression = make_pipeline(
            RobustScaler(),
            LogisticRegression()
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = pipeline_linear_regression.fit(X=X_train, y=y_train)

        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{futur_obj.mlflow_model_name}.pickle'
        pickle.dump(model, open(model_path, 'wb'))

        additional_parameters = {
            'dataset': 'sample_dataset_10000_enriched',
            'using_BERT_masking_predictions': 'pourcentage_of_correct_prediction',
            'sample_dataset':'sample_dataset_10000_enriched'
        }

        # mlflow_save_params
        mlflow_save_params(
            training_set_size= X_test.shape[0],
            row_count= big_df.shape[0],
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
        example_df = big_df.sample(3)

        mlflow_save_model(
            model=model,
            is_tensorflow=False,
            model_name=futur_obj.mlflow_model_name,
            input_example=example_df
        )

        mlflow.end_run()
