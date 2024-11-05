
import os
import pandas as pd
import numpy as np

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
