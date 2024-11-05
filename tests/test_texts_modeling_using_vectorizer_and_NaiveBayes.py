
import unittest
from numpy.testing import assert_almost_equal

import pandas as pd
from sklearn.model_selection import train_test_split

from detect_ai_content.ml_logic.for_texts.using_vectorizer_and_NaiveBayes.using_vectorizer_and_NaiveBayes import *

class TestTextModelingVectorizer(unittest.TestCase):
    def test_training(self):
        """
        Test the method to Train & validate the model
        """
        path = "./tests/data/sample_dataset_1000.csv"
        df = pd.read_csv(path)
        X = df['text']
        y = df['generated']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = train_model(X_train, y_train)
        results = evaluate_model(model=model, X_test=X_test, y_test=y_test)
        assert_almost_equal(results["accuracy_score"], 0.9, decimal=1)
        assert_almost_equal(results["f1_score"], 0.9, decimal=1)
        assert_almost_equal(results["precision_score"], 0.9, decimal=1)
        assert_almost_equal(results["recall_score"], 0.9, decimal=1)

if __name__ == '__main__':
    unittest.main()
