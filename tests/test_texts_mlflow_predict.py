
import unittest
import mlflow
import pandas as pd

from detect_ai_content.ml_logic.mlflow import load_model
from detect_ai_content.params import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import preprocess, evaluate_model

class TestMLFlowTextPrediction(unittest.TestCase):
    def test_load_text_model_FE(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = load_model(model_name=MLFLOW_FE_MODEL_NAME, is_tensorflow=False, stage="Production")
        self.assertEqual(model is not None, True)

    def test_load_text_model_Vectorizer(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = load_model(model_name=MLFLOW_VECTORIZER_MODEL_NAME, is_tensorflow=False, stage="Production")
        self.assertEqual(model is not None, True)

    def test_mlflow_model_FE_predictions(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = load_model(model_name=MLFLOW_FE_MODEL_NAME, is_tensorflow=False, stage="Production")

        path = "./tests/data/sample_dataset_1000.csv"
        df = pd.read_csv(path)
        X = df[['text']]
        y = df['generated']

        X_preprocessed = preprocess(X)
        results = evaluate_model(model=model, X_test_processed=X_preprocessed, y_test=y)
        self.assertGreater(results["recall_score"], 0.5, "Model prediction should be > to 0.5")
        self.assertGreater(results["f1_score"], 0.5, "Model prediction should be > to 0.5")
        self.assertGreater(results["precision_score"], 0.5, "Model prediction should be > to 0.5")
        self.assertGreater(results["accuracy_score"], 0.5, "Model prediction should be > to 0.5")

if __name__ == '__main__':
    unittest.main()
