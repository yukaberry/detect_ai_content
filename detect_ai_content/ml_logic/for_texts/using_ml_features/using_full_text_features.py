
from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

def train_LogisticRegression_model(X_train_processed, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_processed, y_train)
    return model

def evaluate_model(model, X_test_processed, y_test):
    y_pred = model.predict(X_test_processed)

    return {
        "accuracy_score": accuracy_score(y_true=y_test, y_pred=y_pred),
        "f1_score" : f1_score(y_true=y_test, y_pred=y_pred),
        "precision_score": precision_score(y_true=y_test, y_pred=y_pred),
        "recall_score" : recall_score(y_true=y_test, y_pred=y_pred)
    }

def enrich(data):
    data_enriched = data.copy()
    print('enrich compute_punctuation_in_text')
    data_enriched['punctuations_nb'] = data_enriched['text'].apply(compute_punctuation_in_text)

    print('enrich compute_neg_sentiment_polarity_in_text')
    data_enriched['neg_sentiment_polarity'] = data_enriched['text'].apply(compute_neg_sentiment_polarity_in_text)

    print('enrich compute_pos_sentiment_polarity_in_text')
    data_enriched['pos_sentiment_polarity'] = data_enriched['text'].apply(compute_pos_sentiment_polarity_in_text)

    print('enrich text_corrections')
    # Temp deactivated because it's very long to compute .... but very interesting to keep
    data_enriched['text_corrections_nb'] = data_enriched['text'].apply(compute_number_of_text_corrections)

    print('enrich compute_repetitions_in_text')
    data_enriched['text_repetitions_nb'] = data_enriched['text'].apply(compute_repetitions_in_text)

    print('enrich number_of_sentences')
    data_enriched['number_of_sentences'] = data_enriched['text'].apply(number_of_sentences)

    print('enrich text_lenght')
    data_enriched['text_lenght'] = data_enriched['text'].apply(text_lenght)

    return data_enriched

def preprocess(data, auto_enrich=True):
    if auto_enrich:
        data_processed = enrich(data=data)
    else:
        data_processed = data

    data_processed['average_punctuations_by_sentence'] = data_processed['punctuations_nb'] / data_processed['number_of_sentences']
    data_processed['average_sentence_lenght'] = data_processed['text_lenght'] / data_processed['number_of_sentences']
    data_processed['average_text_corrections_by_sentence'] = data_processed['text_corrections_nb'] / data_processed['number_of_sentences']
    data_processed['average_text_corrections_by_lenght'] = data_processed['text_corrections_nb'] / data_processed['text_lenght']
    data_processed['average_text_repetitions_by_sentence'] = data_processed['text_repetitions_nb'] / data_processed['number_of_sentences']
    data_processed['average_text_repetitions_by_lenght'] = data_processed['text_repetitions_nb'] / data_processed['text_lenght']
    data_processed['average_neg_sentiment_polarity_by_sentence'] = data_processed['neg_sentiment_polarity'] / data_processed['number_of_sentences']
    data_processed['average_pos_sentiment_polarity_by_sentence'] = data_processed['pos_sentiment_polarity'] / data_processed['number_of_sentences']

    data_processed = data_processed[[
        'average_punctuations_by_sentence',
        'average_sentence_lenght',
        'average_text_corrections_by_sentence',
        'average_text_corrections_by_lenght',
        'average_text_repetitions_by_sentence',
        'average_text_repetitions_by_lenght',
        'average_neg_sentiment_polarity_by_sentence',
        'average_pos_sentiment_polarity_by_sentence'
    ]]

    scaler = RobustScaler()
    return scaler.fit_transform(data_processed)
