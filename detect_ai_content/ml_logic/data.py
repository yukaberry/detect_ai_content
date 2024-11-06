
import os
import pandas as pd
import numpy as np

from detect_ai_content.ml_logic.for_texts.using_ml_features.features.features import *

def get_enriched_df(size = None):
    import detect_ai_content
    module_dir_path = os.path.dirname(detect_ai_content.__file__)
    df = pd.read_csv(f'{module_dir_path}/../raw_data/huggingface.co_human_ai_generated_text/model_training_dataset_enriched.csv')

    df['repetitions_ratio'] = df['text_repetitions_nb']/df['text_lenght']
    df['punctuations_ratio'] = df['punctuations_nb']/df['text_lenght']
    df['text_corrections_ratio'] = df['text_corrections_nb']/df['text_lenght']
    df['text_corrections_set_ratio'] = df['text_corrections_nb']/df['number_of_sentences']
    df['average_neg_sentiment_polarity'] = df['neg_sentiment_polarity']/df['text_lenght']

    AI_Human_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/kaggle-ai-generated-vs-human-text/AI_Human_enriched.csv')
    AI_Human_enriched_df['repetitions_ratio'] = AI_Human_enriched_df['text_repetitions_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['punctuations_ratio'] = AI_Human_enriched_df['punctuations_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['text_corrections_ratio'] = AI_Human_enriched_df['text_corrections_nb']/AI_Human_enriched_df['text_lenght']
    AI_Human_enriched_df['text_corrections_set_ratio'] = AI_Human_enriched_df['text_corrections_nb']/AI_Human_enriched_df['number_of_sentences']
    AI_Human_enriched_df['average_sentence_lenght'] = AI_Human_enriched_df['text_lenght']/AI_Human_enriched_df['number_of_sentences']
    AI_Human_enriched_df['average_neg_sentiment_polarity'] = AI_Human_enriched_df['neg_sentiment_polarity']/AI_Human_enriched_df['text_lenght']

    daigt_v2_enriched_df = pd.read_csv(f'{module_dir_path}/../raw_data/daigt-v2-train-dataset/train_v2_drcat_02_enriched.csv')
    daigt_v2_enriched_df['repetitions_ratio'] = daigt_v2_enriched_df['text_repetitions_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['punctuations_ratio'] = daigt_v2_enriched_df['punctuations_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['text_corrections_ratio'] = daigt_v2_enriched_df['text_corrections_nb']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['text_corrections_set_ratio'] = daigt_v2_enriched_df['text_corrections_nb']/daigt_v2_enriched_df['number_of_sentences']
    daigt_v2_enriched_df['average_neg_sentiment_polarity'] = daigt_v2_enriched_df['neg_sentiment_polarity']/daigt_v2_enriched_df['text_lenght']
    daigt_v2_enriched_df['repetitions_ratio'].mean()

    big_df = pd.concat(objs=[df, AI_Human_enriched_df, daigt_v2_enriched_df])
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

    return data_enriched
