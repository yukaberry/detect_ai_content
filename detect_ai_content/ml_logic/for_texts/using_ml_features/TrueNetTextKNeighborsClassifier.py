
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import preprocess, enrich, evaluate_model
from detect_ai_content.ml_logic.data import get_enriched_df

import os
import pandas as pd
import numpy as np
import pickle

class TrueNetTextKNeighborsClassifier:
    def __init__(self):
        self.name = "TrueNetTextKNeighborsClassifier"
        self.description = ""

    def run_grid_search():
        df = get_enriched_df(100)
        X = df[[
            'repetitions_ratio',
            'punctuations_ratio',
            'text_corrections_ratio',
            'average_sentence_lenght',
            'average_neg_sentiment_polarity',
        ]]
        y = df['generated']

        pipeline = make_pipeline(
            RobustScaler(),
            KNeighborsClassifier()
        )

        k_range = list(range(1, 31))
        param_grid = dict(kneighborsclassifier__n_neighbors=k_range)

        grid = GridSearchCV(pipeline ,param_grid,refit = True, verbose=2)
        grid.fit(X,y)
        print(grid.best_estimator_)

    def retrain_full_model():
        big_df = get_enriched_df(10_000)

        pipeline = make_pipeline(
            RobustScaler(),
            KNeighborsClassifier(n_neighbors=20) # param from run_grid_search
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

        results = evaluate_model(model=model, X_test_processed=X_test, y_test=y_test)
        print(results)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/TrueNetTextKNeighborsClassifier.pickle'
        pickle.dump(model, open(model_path, 'wb'))
