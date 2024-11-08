
import unittest
import pandas as pd

from detect_ai_content.params import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import TrueNetTextLogisticRegression
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import TrueNetTextTfidfNaiveBayesClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import TrueNetTextDecisionTreeClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import TrueNetTextSVC
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import TrueNetTextKNeighborsClassifier
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextUsingBERTMaskedPredictions import TrueNetTextUsingBERTMaskedPredictions

from detect_ai_content.ml_logic.preprocess import preprocess
from detect_ai_content.ml_logic.data import get_enriched_df
from detect_ai_content.ml_logic.evaluation import evaluate_model

class TestMLFlowTextPrediction(unittest.TestCase):
    def test_mlflow_predictions(self):
        df = get_enriched_df(purpose="test")
        y_test = df['generated']
        X_test_processed = preprocess(data=df, auto_enrich=False)

        TrueNetTextLogisticRegression_model = TrueNetTextLogisticRegression()._load_model(stage="staging")
        TrueNetTextLogisticRegression_results = evaluate_model(model=TrueNetTextLogisticRegression_model, X_test_processed=X_test_processed, y_test=y_test)
        print(TrueNetTextLogisticRegression_results)

        TrueNetTextDecisionTreeClassifier_model = TrueNetTextDecisionTreeClassifier()._load_model(stage="staging")
        TrueNetTextDecisionTreeClassifier_results = evaluate_model(model=TrueNetTextDecisionTreeClassifier_model, X_test_processed=X_test_processed, y_test=y_test)
        print(TrueNetTextDecisionTreeClassifier_results)

        TrueNetTextKNeighborsClassifier_model = TrueNetTextKNeighborsClassifier()._load_model(stage="staging")
        TrueNetTextKNeighborsClassifier_results = evaluate_model(model=TrueNetTextKNeighborsClassifier_model, X_test_processed=X_test_processed, y_test=y_test)
        print(TrueNetTextKNeighborsClassifier_results)

        TrueNetTextSVC_model = TrueNetTextSVC()._load_model(stage="staging")
        TrueNetTextSVC_results = evaluate_model(model=TrueNetTextSVC_model, X_test_processed=X_test_processed, y_test=y_test)
        print(TrueNetTextSVC_results)

        TrueNetTextUsingBERTMaskedPredictions_model = TrueNetTextUsingBERTMaskedPredictions()._load_model(stage="staging")
        X_processed_for_BERT = TrueNetTextUsingBERTMaskedPredictions.preprocess(data=df)
        TrueNetTextUsingBERTMaskedPredictions_results = evaluate_model(model=TrueNetTextUsingBERTMaskedPredictions_model, X_test_processed=X_processed_for_BERT, y_test=y_test)
        print(TrueNetTextUsingBERTMaskedPredictions_results)

        TrueNetTextTfidfNaiveBayesClassifier_model = TrueNetTextTfidfNaiveBayesClassifier()._load_model(stage="Staging")
        TrueNetTextTfidfNaiveBayesClassifier_results = evaluate_model(model=TrueNetTextTfidfNaiveBayesClassifier_model, X_test_processed=df['text'], y_test=y_test)
        print(TrueNetTextTfidfNaiveBayesClassifier_results)


if __name__ == '__main__':
    unittest.main()
