

from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model
from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model
from detect_ai_content.ml_logic.preprocess import smartCleanerTransformer, smartSelectionTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import os
import mlflow
from mlflow import MlflowClient
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

class TrueNetTextUsingBERTMaskedPredictions:
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
        self.model = self.local_trained_pipeline()

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


    def retrain_production_pipeline():
        columns = [
            'pourcentage_of_correct_prediction'
        ]

        features_selection_transformer = smartSelectionTransformer(columns=columns)
        pipeline = Pipeline([
            ('row_cleaner', smartCleanerTransformer()),
            ('enricher', smartBertEnrichTransformer()),
            ('features_selection', features_selection_transformer),
            ('scaler', RobustScaler()),
            ('estimator', LogisticRegression()),
             ])

        df = get_enriched_df()
        y = df['generated']

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, )
        pipeline.fit(X=X_train, y=y_train)

        results = evaluate_model(pipeline, X_test, y_test)
        print(results)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        mlflow_model_name = TrueNetTextUsingBERTMaskedPredictions().mlflow_model_name
        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{mlflow_model_name}_pipeline.pickle'
        pickle.dump(pipeline, open(model_path, 'wb'))


from sklearn.preprocessing import FunctionTransformer

def smartBertEnrichFunction(data):
    '''
        Create features if they don't exist
    '''
    data_processed = data.copy()
    if "pourcentage_of_correct_prediction" not in data_processed:
        data_processed = enrich_text_BERT_predictions(data_processed)
    return data_processed

def smartBertEnrichTransformer():
    return FunctionTransformer(smartBertEnrichFunction)

import string
import torch
from textblob import TextBlob

print(f'torch.cuda.is_available:{torch.cuda.is_available()}')

from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

import platform
if platform.system() == 'Darwin':
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'device:{device}')
bert_model.to(device)

import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
list_physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(f'tf.config.experimental.list_physical_devices(GPU):{list_physical_devices}')


def compute_masked_words_BERT_prediction(text):
    text_blob = TextBlob(text)
    number_of_test = 0
    number_of_correct_prediction = 0

    for sentence in text_blob.sentences:
        if len(sentence) > 500:
            # print('ignore the sentence')
            continue

        for word in sentence.words:
            if len(word) > 5:
                masked_sentence = sentence.replace(word, "<mask>")
                # print(f'START_{masked_sentence}_END')
                top_k = 10
                top_clean = 5
                input_ids, mask_idx = BERT_encode(bert_tokenizer, f'{masked_sentence}')
                with torch.no_grad():
                    predict = bert_model(input_ids)[0]
                predict_words = BERT_decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
                # print(predict_words)

                number_of_test += 1
                if word in predict_words:
                    number_of_correct_prediction +=1

    # print(f"number_of_test: {number_of_test}")
    # print(f"number_of_correct_prediction: {number_of_correct_prediction}")
    # print(f"prediction : {round(100 * number_of_correct_prediction/number_of_test)}%")
    return (number_of_test, number_of_correct_prediction)

def BERT_decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def BERT_encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)]).to(device)
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def enrich_text_BERT_predictions(data):
    print('enrich_text_BERT_predictions')
    data_enriched = data.copy()

    pourcentage_of_correct_predictions = []
    number_of_tests = []
    number_of_correct_predictions = []

    index_sum = 0
    for (index, row) in data_enriched.iterrows():
        text = row['text']
        index_sum += 1
        (number_of_test, number_of_correct_prediction) = compute_masked_words_BERT_prediction(text)
        pourcentage = -1
        if number_of_test > 0:
            pourcentage = round(100 * number_of_correct_prediction/number_of_test)

        pourcentage_of_correct_predictions.append(pourcentage)
        number_of_tests.append(number_of_test)
        number_of_correct_predictions.append(number_of_correct_prediction)
        print(index_sum)

    data_enriched['number_of_tests'] = number_of_tests
    data_enriched['number_of_correct_prediction'] = number_of_correct_predictions
    data_enriched['pourcentage_of_correct_prediction'] = pourcentage_of_correct_predictions
    return data_enriched
