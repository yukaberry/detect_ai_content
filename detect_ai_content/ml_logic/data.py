
import os
import pandas as pd
import numpy as np

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import *

from nltk.tokenize import word_tokenize, sent_tokenize
import textstat


def get_enriched_df(purpose="train", size = None):
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)
    df = pd.read_csv(f'{module_dir_path}/../raw_data/texts_merged_dataset_enriched/texts_merged_dataset_enriched_{purpose}.csv')
    big_df = df
    big_df = big_df[np.isinf(big_df['average_sentence_lenght']) == False]
    big_df = big_df[np.isinf(big_df['repetitions_ratio']) == False]
    big_df = big_df[np.isinf(big_df['punctuations_ratio']) == False]
    big_df = big_df[np.isinf(big_df['text_corrections_ratio']) == False]
    big_df = big_df[np.isinf(big_df['average_neg_sentiment_polarity']) == False]
    big_df = big_df.fillna(0)

    if size is not None:
        return big_df.sample(size)

    return big_df


def enrich_text(data):
    data_enriched = data.copy()
    # print('enrich compute_punctuation_in_text')
    data_enriched['punctuations_nb'] = data_enriched['text'].apply(compute_punctuation_in_text)

    # print('enrich compute_neg_sentiment_polarity_in_text')
    data_enriched['neg_sentiment_polarity'] = data_enriched['text'].apply(compute_neg_sentiment_polarity_in_text)

    # print('enrich compute_pos_sentiment_polarity_in_text')
    data_enriched['pos_sentiment_polarity'] = data_enriched['text'].apply(compute_pos_sentiment_polarity_in_text)

    # print('enrich text_corrections')
    data_enriched['text_corrections_nb'] = data_enriched['text'].apply(compute_number_of_text_corrections_using_nltk_words)

    # print('enrich compute_repetitions_in_text')
    data_enriched['text_repetitions_nb'] = data_enriched['text'].apply(compute_repetitions_in_text)

    # print('enrich number_of_sentences')
    data_enriched['number_of_sentences'] = data_enriched['text'].apply(number_of_sentences)

    # print('enrich text_lenght')
    data_enriched['text_lenght'] = data_enriched['text'].apply(text_lenght)

    data_enriched['repetitions_ratio'] = data_enriched['text_repetitions_nb']/data_enriched['text_lenght']
    data_enriched['punctuations_ratio'] = data_enriched['punctuations_nb']/data_enriched['text_lenght']
    data_enriched['text_corrections_ratio'] = data_enriched['text_corrections_nb']/data_enriched['text_lenght']
    data_enriched['text_corrections_set_ratio'] = data_enriched['text_corrections_nb']/data_enriched['number_of_sentences']
    data_enriched['average_neg_sentiment_polarity'] = data_enriched['neg_sentiment_polarity']/data_enriched['text_lenght']
    data_enriched['average_pos_sentiment_polarity'] = data_enriched['pos_sentiment_polarity']/data_enriched['text_lenght']
    data_enriched['average_sentence_lenght'] = data_enriched['text_lenght']/data_enriched['number_of_sentences']

    data_enriched = data_enriched[np.isinf(data_enriched['average_sentence_lenght']) == False]
    data_enriched = data_enriched[np.isinf(data_enriched['repetitions_ratio']) == False]
    data_enriched = data_enriched[np.isinf(data_enriched['punctuations_ratio']) == False]
    data_enriched = data_enriched[np.isinf(data_enriched['text_corrections_ratio']) == False]
    data_enriched = data_enriched[np.isinf(data_enriched['average_neg_sentiment_polarity']) == False]
    data_enriched = data_enriched[np.isinf(data_enriched['average_pos_sentiment_polarity']) == False]

    data_enriched = data_enriched.dropna()

    return data_enriched

def enrich_text_BERT_predictions(data):
    print('enrich_text_BERT_predictions')
    data_enriched = data.copy()

    pourcentage_of_correct_predictions = []
    number_of_tests = []
    number_of_correct_predictions = []

    index_sum = 0
    for (index, row) in data_enriched.iterrows():
        text = row['text']
        index_sum += 1
        (number_of_test, number_of_correct_prediction) = compute_masked_words_BERT_prediction(text)
        pourcentage = -1
        if number_of_test > 0:
            pourcentage = round(100 * number_of_correct_prediction/number_of_test)

        pourcentage_of_correct_predictions.append(pourcentage)
        number_of_tests.append(number_of_test)
        number_of_correct_predictions.append(number_of_correct_prediction)
        print(index_sum)

    data_enriched['number_of_tests'] = number_of_tests
    data_enriched['number_of_correct_prediction'] = number_of_correct_predictions
    data_enriched['pourcentage_of_correct_prediction'] = pourcentage_of_correct_predictions
    return data_enriched

def enrich_lexical_diversity_readability(data):
    enriched_data = data.copy()

    # Lexical diversity (Unique words / Total words)
    enriched_data['lexical_diversity'] = data['text'].apply(lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)) if len(word_tokenize(x)) > 0 else 0)

    # Readability Scores
    enriched_data['flesch_reading_ease'] = data['text'].apply(textstat.flesch_reading_ease)
    enriched_data['smog_index'] = data['text'].apply(textstat.smog_index)
    enriched_data['flesch_kincaid_grade'] = data['text'].apply(textstat.flesch_kincaid_grade)

    # Some metrics
    enriched_data['avg_word_length'] = data['text'].apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]))

    return enriched_data
