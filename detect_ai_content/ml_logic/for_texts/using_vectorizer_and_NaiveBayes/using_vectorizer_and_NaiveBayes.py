
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_model(X, y):
    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    model = pipeline_naive_bayes.fit(X=X['text'].values, y=y.values)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test['text'].values)

    return {
        "accuracy_score": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_score" : f1_score(y_true=y_test, y_pred=y_pred),
        "precision_score": precision_score(y_true=y_test, y_pred=y_pred),
        "recall_score" : recall_score(y_true=y_test, y_pred=y_pred)
    }
