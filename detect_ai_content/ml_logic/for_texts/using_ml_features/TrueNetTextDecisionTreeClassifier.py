
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import preprocess, enrich, evaluate_model
from detect_ai_content.ml_logic.data import get_enriched_df

import os
import pickle

class TrueNetTextDecisionTreeClassifier:
    def __init__(self):
        self.name = "TrueNetTextDecisionTreeClassifier"
        self.description = ""

    def run_grid_search():
        param_grid = {
            'C':[1,10,100,1000],
            'gamma':[1,0.1,0.001,0.0001],
            'kernel':['linear','rbf']
        }

        df = get_enriched_df(10_000)
        X = df[[
            'repetitions_ratio',
            'punctuations_ratio',
            'text_corrections_ratio',
            'average_sentence_lenght',
            'average_neg_sentiment_polarity',
        ]]
        y = df['generated']
        grid = GridSearchCV(DecisionTreeClassifier(),param_grid,refit = True, verbose=2)
        grid.fit(X,y)
        print(grid.best_estimator_)

    def retrain_full_model():
        big_df = get_enriched_df()

        pipeline = make_pipeline(
            RobustScaler(),
            DecisionTreeClassifier()
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
        model_path = f = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/DecisionTreeClassifier.pickle'
        pickle.dump(model, open(model_path, 'wb'))
