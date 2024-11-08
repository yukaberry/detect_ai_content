
import unittest
import pandas as pd

from detect_ai_content.params import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextLogisticRegression import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextTfidfNaiveBayesClassifier import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextDecisionTreeClassifier import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextSVC import *
from detect_ai_content.ml_logic.for_texts.using_ml_features.TrueNetTextKNeighborsClassifier import *

from detect_ai_content.ml_logic.preprocess import preprocess
from detect_ai_content.ml_logic.data import get_enriched_df

class TestMLFlowTextPrediction(unittest.TestCase):
    def test_mlflow_predictions(self):
        df = get_enriched_df(purpose="test")
        y_test = df['generated']
        X_test_processed = preprocess(data=df, auto_enrich=False)

        TrueNetTextLogisticRegression_model = TrueNetTextLogisticRegression()._load_model(stage="staging")
        TrueNetTextLogisticRegression_results = evaluate_model(model=TrueNetTextLogisticRegression_model, X_test_processed=X_test_processed, y_test=y_test)
        print(TrueNetTextLogisticRegression_results)
        return

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
