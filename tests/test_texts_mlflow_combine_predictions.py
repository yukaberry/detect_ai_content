
import unittest
import pandas as pd

from detect_ai_content.params import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import *

from detect_ai_content.ml_logic.preprocess import preprocess

class TestMLFlowTextPrediction(unittest.TestCase):
    def test_mlflow_predictions(self):
        path = "../detect_ai_content/raw_data/samples/sample_dataset_1000.csv"
        df = pd.read_csv(path)

        TrueNetTextLogisticRegression_model = TrueNetTextLogisticRegression().model
        TrueNetTextTfidfNaiveBayesClassifier_model = TrueNetTextTfidfNaiveBayesClassifier().model
        TrueNetTextDecisionTreeClassifier_model = TrueNetTextDecisionTreeClassifier().model
        TrueNetTextSVC_model = TrueNetTextSVC().model
        TrueNetTextKNeighborsClassifier_model = TrueNetTextKNeighborsClassifier().model

        X = df[['text']]
        y = df['generated']

        X_preprocessed = preprocess(X)

        TrueNetTextLogisticRegression_results = evaluate_model(model=TrueNetTextLogisticRegression_model, X_test_processed=X_preprocessed, y_test=y)
        TrueNetTextLogisticRegression_results['model'] = 'TrueNetTextLogisticRegression'

        TrueNetTextTfidfNaiveBayesClassifier_model_results = evaluate_model(model=TrueNetTextTfidfNaiveBayesClassifier_model, X_test_processed=df['text'], y_test=y)
        TrueNetTextTfidfNaiveBayesClassifier_model_results['model'] = 'TrueNetTextTfidfNaiveBayesClassifier'

        TrueNetTextDecisionTreeClassifier_results = evaluate_model(model=TrueNetTextDecisionTreeClassifier_model, X_test_processed=X_preprocessed, y_test=y)
        TrueNetTextDecisionTreeClassifier_results['model'] = 'TrueNetTextDecisionTreeClassifier'

        TrueNetTextSVC_results = evaluate_model(model=TrueNetTextSVC_model, X_test_processed=X_preprocessed, y_test=y)
        TrueNetTextSVC_results['model'] = 'TrueNetTextSVC'

        TrueNetTextKNeighborsClassifier_results = evaluate_model(model=TrueNetTextKNeighborsClassifier_model, X_test_processed=X_preprocessed, y_test=y)
        TrueNetTextKNeighborsClassifier_results['model'] = 'TrueNetTextKNeighborsClassifier'

        df = pd.DataFrame(data=[
            TrueNetTextLogisticRegression_results,
            TrueNetTextTfidfNaiveBayesClassifier_model_results,
            TrueNetTextDecisionTreeClassifier_results,
            TrueNetTextSVC_results,
            TrueNetTextKNeighborsClassifier_results
        ])

        print(df)


if __name__ == '__main__':
    unittest.main()
