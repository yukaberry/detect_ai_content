
import unittest
import pandas as pd

from detect_ai_content.ml_logic.mlflow import load_model
from detect_ai_content.params import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import *
from detect_ai_content.ml_logic.preprocess import preprocess

class TestMLFlowTextPrediction(unittest.TestCase):
    def test_load_text_model_FE(self):
        model = TrueNetTextLogisticRegression().model
        self.assertEqual(model is not None, True)

    def test_load_text_model_Vectorizer(self):
        model = TrueNetTextTfidfNaiveBayesClassifier().model
        self.assertEqual(model is not None, True)

    def test_mlflow_model_FE_predictions(self):
        model = TrueNetTextLogisticRegression().model
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
