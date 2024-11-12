
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

        TrueNetTextLogisticRegression_model = TrueNetTextLogisticRegression().model
        TrueNetTextLogisticRegression_results = evaluate_model(model=TrueNetTextLogisticRegression_model, X_test_processed=df, y_test=y_test)
        print(TrueNetTextLogisticRegression_results)

        TrueNetTextDecisionTreeClassifier_model = TrueNetTextDecisionTreeClassifier().local_trained_pipeline()
        TrueNetTextDecisionTreeClassifier_results = evaluate_model(model=TrueNetTextDecisionTreeClassifier_model, X_test_processed=df, y_test=y_test)
        print(TrueNetTextDecisionTreeClassifier_results)

        TrueNetTextKNeighborsClassifier_model = TrueNetTextKNeighborsClassifier().local_trained_pipeline()
        TrueNetTextKNeighborsClassifier_results = evaluate_model(model=TrueNetTextKNeighborsClassifier_model, X_test_processed=df, y_test=y_test)
        print(TrueNetTextKNeighborsClassifier_results)

        TrueNetTextSVC_model = TrueNetTextSVC().local_trained_pipeline()
        TrueNetTextSVC_results = evaluate_model(model=TrueNetTextSVC_model, X_test_processed=df, y_test=y_test)
        print(TrueNetTextSVC_results)

        TrueNetTextUsingBERTMaskedPredictions_model = TrueNetTextUsingBERTMaskedPredictions().local_trained_pipeline()
        TrueNetTextUsingBERTMaskedPredictions_results = evaluate_model(model=TrueNetTextUsingBERTMaskedPredictions_model, X_test_processed=df, y_test=y_test)
        print(TrueNetTextUsingBERTMaskedPredictions_results)

        TrueNetTextTfidfNaiveBayesClassifier_model = TrueNetTextTfidfNaiveBayesClassifier().local_trained_pipeline()
        TrueNetTextTfidfNaiveBayesClassifier_results = evaluate_model(model=TrueNetTextTfidfNaiveBayesClassifier_model,X_test_processed=df, y_test=y_test)
        print(TrueNetTextTfidfNaiveBayesClassifier_results)


if __name__ == '__main__':
    unittest.main()
