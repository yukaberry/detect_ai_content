import pandas as pd
import numpy as np
import re
from collections import Counter

# Text and NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import textstat
from textblob import TextBlob
from spellchecker import SpellChecker

from transformers import pipeline

# Load Spacy model for NLP
nlp = spacy.load("en_core_web_sm")

import nltk
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
# Load SpaCy's English model with only NER (for named entity recognition)
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
import os
import json
import torch
import platform

class InternalFeatures():

    def __init__(self ):

        # Open and load the JSON file
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        with open(f'{module_dir_path}/ml_logic/for_texts/slang_dict.json', 'r') as file:
            slang_dictonary = json.load(file)

        self.slang_dict = slang_dictonary


    def text_based_features(self, text_df, text_column='text'):
        print("text_based_features")

        # TODO
        # not to use df type -->> numoy arry
        features = pd.DataFrame()

        # Ensure stopwords are loaded
        stop_words = set(stopwords.words('english'))

        # Word count
        features['word_count'] = text_df[text_column].apply(lambda x: len(word_tokenize(x)))

        # Sentence count
        features['sentence_count'] = text_df[text_column].apply(lambda x: len(sent_tokenize(x)))

        # Average word length
        features['avg_word_length'] = text_df[text_column].apply(lambda x: np.mean([len(word) for word in word_tokenize(x)]))

        # Stopwords count
        features['stopwords_count'] = text_df[text_column].apply(lambda x: sum(1 for word in word_tokenize(x) if word.lower() in stop_words))

        return features


    def lexical_diversity_readability(self, text_df, text_column='text'):
        print("lexical_diversity_readability")

        # TODO
        # not to use df type -->> numoy arry
        features = pd.DataFrame()

        # Lexical diversity (Unique words / Total words)
        features['lexical_diversity'] = text_df[text_column].apply(lambda x: len(set(word_tokenize(x))) / len(word_tokenize(x)) if len(word_tokenize(x)) > 0 else 0)

        # Readability Scores
        features['flesch_reading_ease'] = text_df[text_column].apply(textstat.flesch_reading_ease)
        features['smog_index'] = text_df[text_column].apply(textstat.smog_index)
        features['flesch_kincaid_grade'] = text_df[text_column].apply(textstat.flesch_kincaid_grade)

        return features


    def pos_tagging_features(self, text_df, text_column='text'):
        print("pos_tagging_features")

        # TODO
        # change datatype
        pos_features = pd.DataFrame()

        def pos_counts(text):
            doc = nlp(text) # text parsing
            pos_counts = {token.pos_: 0 for token in doc} # dictionary with POS tags (like NOUN, VERB, ADJ)
            for token in doc:
                pos_counts[token.pos_] += 1
            return pos_counts

        pos_df = text_df[text_column].apply(pos_counts).apply(pd.Series).fillna(0) # Fill missing POS tag counts with 0 for consistency across texts

        pos_df.columns = [f'pos_{col}' for col in pos_df.columns]
        pos_features = pd.concat([pos_features, pos_df], axis=1)

        return pos_features


    def sentiment_emotion_features(self, text_df, text_column='text'):
        print("sentiment_emotion_features")

        # TODO
        # change data type

        # TODO
        # COMPUTATION SUPER HEAVY!!!

        features = pd.DataFrame()

        # Initialize sentiment analysis pipeline with GPU (device=0)
        # TODO
        # local test run (device=1)
        device = torch.device("cpu")
        if platform.system() == 'Darwin':
            device = torch.device("mps")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        sentiment_pipeline = pipeline('sentiment-analysis',
                                      device=device,
                                      truncation=True,
                                      max_length=512)
        print(sentiment_pipeline)

        # Apply sentiment analysis in batches
        texts = text_df[text_column].tolist()
        print(texts)

        sentiment_results = sentiment_pipeline(texts, truncation=True, max_length=512, batch_size=32)
        print(sentiment_results)

        # Extract sentiment labels from the results
        features['sentiment'] = [result['label'] for result in sentiment_results]

        # Polarity and subjectivity from TextBlob
        features['polarity'] = text_df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
        features['subjectivity'] = text_df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)

        return features


    def ngrams_keyword_features(self, text_df, text_column='text'):
        print("ngrams_keyword_features")

        #TODO
        # change datatype
        features = pd.DataFrame()

        # Bi-grams count
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        bigram_matrix = vectorizer.fit_transform(text_df[text_column])
        features['bigram_count'] = bigram_matrix.sum(axis=1).A1

        # Trigrams count
        vectorizer = CountVectorizer(ngram_range=(3, 3))
        trigram_matrix = vectorizer.fit_transform(text_df[text_column])
        features['trigram_count'] = trigram_matrix.sum(axis=1).A1 # .A1 converts the sparse column vector into a 1-d np array, to work with pd df columns.

        return features


    def linguistic_complexity_features(self, text_df, text_column='text'):
        print("linguistic_complexity_features")

        # TODO
        # datatype change
        features = pd.DataFrame()

        # Process text in batches to improve efficiency
        docs = nlp.pipe(text_df[text_column], batch_size=32)  # Adjust batch size as needed for performance
        features['dependency_count'] = [len(doc.ents) for doc in docs]

        return features


    def spelling_error_features(self, text_df, text_column='text'):
        print("spelling_error_features")

        # TODO
        # Datatype change
        features = pd.DataFrame()

        spell = SpellChecker()

        def count_spelling_errors(text):
            words = text.split()
            misspelled_words = spell.unknown(words)
            return len(misspelled_words)

        features['spelling_errors'] = text_df[text_column].apply(count_spelling_errors)
        return features


    def repetition_features(self, text_df, text_column='text', n=2):
        print("repetition_features")

        #TODO
        # CHANGE dataype
        features = pd.DataFrame()

        def detect_repetition(text):
            words = text.split()
            phrases = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
            phrase_counts = Counter(phrases)
            repeated_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}
            return len(repeated_phrases)  # count of unique repeated phrases

        features['repetition_count'] = text_df[text_column].apply(detect_repetition)

        return features


    def structural_formatting_features(self, text_df, text_column='text'):
        print("structural_formatting_features")

        # TODO
        # change datatype
        features = pd.DataFrame()

        # Line break count
        features['line_break_count'] = text_df[text_column].apply(lambda x: x.count('\n'))

        # Punctuation count
        features['punctuation_count'] = text_df[text_column].apply(lambda x: sum([1 for char in x if char in '.,;!?']))

        return features


    def count_slang(self, raw_df, slang_dictobary):
        print("count_slang")

        # TODO
        # change datatype
        features = pd.DataFrame()

        def preprocess_text(text):
            # Lowercase, remove punctuation
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            words = text.split()
            return words

        # text = preprocess_text(text)
        # slang_count = sum(1 for word in text if word in slang_dict)

        # Apply preprocessing and slang count to each row in the specified column
        features['slang_count'] = raw_df['text'].apply(
            lambda text: sum(1 for word in preprocess_text(text) if word in slang_dictobary)
        )

        return features


    def concat_features(self, raw_data, *features):
        print("concat_features")

        """
        - return all of dataframes

        deatils of dataframes : - text_features,
                                - lexical_features,
                                - pos_features,
                                - sentiment_features,
                                - ngram_features,
                                - complexity_features,
                                - structural_features,
                                - repetition_features,
                                - spelling_error_features
                                - slang

        argument : *features
        make this function flexible for any number of DataFrames

        """
        internal_features = pd.concat(features, axis=1)

        # for training
        # internal_df = pd.concat([raw_data[['generated', 'text']],
        #                          internal_features],
        #                         axis=1)

        # for user input 'genetrated' is not nessesary
        internal_df = pd.concat([raw_data,
                                internal_features],
                            axis=1)

        # test
        print(internal_df.shape)

        return internal_df


    def aggregate_internal_features(self, df):

        # # Retain specific features directly
        # retained_features = df[['avg_word_length', 'lexical_diversity',
        #                         'flesch_reading_ease','smog_index',
        #                         'flesch_kincaid_grade', 'polarity',
        #                         'subjectivity', 'slang_count']].copy()

        # # Useful ratios
        # df['word_count_ratio'] = df['word_count'] / df['sentence_count']
        # df['stopwords_ratio'] = df['stopwords_count'] / df['word_count']
        # df['punctuation_ratio'] = df['punctuation_count'] / df['word_count']
        # df['repetition_ratio'] = df['repetition_count'] / df['word_count']
        # df['bigram_count_ratio'] = df['bigram_count'] / df['word_count']
        # df['trigram_count_ratio'] = df['trigram_count'] / df['word_count']
        # df['dependency_ratio'] = df['dependency_count'] / df['sentence_count']
        # df['spelling_errors_ratio'] = df['spelling_errors'] / df['word_count']

        # df = df[['generated',
        #         'word_count_ratio', 'stopwords_ratio','punctuation_ratio',
        #         'repetition_ratio', 'bigram_count_ratio', 'trigram_count_ratio',
        #         'dependency_ratio', 'spelling_errors_ratio', 'sentiment']]

        # # POS tag ratios and any directly included specific features
        # pos_columns = [col for col in df.columns if 'pos_' in col]
        # for pos_col in pos_columns:
        #     df[f'{pos_col}_ratio'] = df[pos_col] / df['word_count']


        # # Encode sentiment feature as numeric values for correlation analysis
        # sentiment_mapping = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}
        # retained_features['sentiment'] = df['sentiment'].map(sentiment_mapping)

        # df = df[['generated',
        #         'word_count_ratio', 'stopwords_ratio','punctuation_ratio',
        #         'repetition_ratio', 'bigram_count_ratio', 'trigram_count_ratio',
        #         'dependency_ratio', 'spelling_errors_ratio']]

        # # Combine all features and include the 'generated' column
        # final_internal_df = pd.concat([df, retained_features],
        #                               axis=1)

        aggregated_features = pd.DataFrame()

        # Useful ratios
        aggregated_features['word_count_ratio'] = df['word_count'] / df['sentence_count']
        aggregated_features['stopwords_ratio'] = df['stopwords_count'] / df['word_count']
        aggregated_features['punctuation_ratio'] = df['punctuation_count'] / df['word_count']
        aggregated_features['repetition_ratio'] = df['repetition_count'] / df['word_count']
        aggregated_features['bigram_count_ratio'] = df['bigram_count'] / df['word_count']
        aggregated_features['trigram_count_ratio'] = df['trigram_count'] / df['word_count']
        aggregated_features['dependency_ratio'] = df['dependency_count'] / df['sentence_count']
        aggregated_features['spelling_errors_ratio'] = df['spelling_errors'] / df['word_count']

        # POS tag ratios and any directly included specific features
        pos_columns = [col for col in df.columns if 'pos_' in col]
        for pos_col in pos_columns:
            aggregated_features[f'{pos_col}_ratio'] = df[pos_col] / df['word_count']

        # Retain specific features directly
        retained_features = df[['avg_word_length', 'lexical_diversity', 'flesch_reading_ease', 'smog_index',
                                        'flesch_kincaid_grade', 'polarity', 'subjectivity']].copy()

        # Encode sentiment feature as numeric values for correlation analysis
        sentiment_mapping = {'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}
        retained_features['sentiment'] = df['sentiment'].map(sentiment_mapping)

        # only for training model
        # Combine all features and include the 'generated' column
        # final_internal_df = pd.concat([df[['generated']], aggregated_features, retained_features], axis=1)

        # for user input
        final_internal_df = pd.concat([aggregated_features, retained_features], axis=1)
        final_internal_df = final_internal_df[['stopwords_ratio', 'punctuation_ratio', 'repetition_ratio',
       'dependency_ratio', 'spelling_errors_ratio', 'pos__ratio',
       'avg_word_length', 'lexical_diversity', 'flesch_reading_ease',
       'smog_index', 'flesch_kincaid_grade', 'sentiment']]

        # TODO
        # ask @linchenpal
        # final_internal_df = final_internal_df.drop(columns=["word_count_ratio", "bigram_count_ratio", "polarity",  "trigram_count_ratio", "subjectivity", "pos_SPACE_ratio" ]) #"pos_SPACE_ratio"

        return final_internal_df

    def process(self, raw_data):
        print("process")
        feature = self.text_based_features(raw_data)
        feature2 = self.lexical_diversity_readability(raw_data)
        feature3 = self.pos_tagging_features(raw_data)
        # TODO
        # feature 4 COMPUTATION SUPER HEAVY!!!
        feature4 = self.sentiment_emotion_features(raw_data)
        feature5 = self.ngrams_keyword_features(raw_data)
        feature6 = self.linguistic_complexity_features(raw_data)
        feature7 = self.spelling_error_features(raw_data)
        feature8 = self.repetition_features(raw_data)
        feature9 = self.structural_formatting_features(raw_data)
        feature10 = self.count_slang(raw_data, self.slang_dict)
        df = self.concat_features(raw_data,
                                    feature, feature2, feature3,feature4,
                                    feature5, feature6, feature7, feature8,
                                    feature9, feature10)

        final_df = self.aggregate_internal_features(df)

        return final_df

# local test
if __name__ == '__main__':

    internal = InternalFeatures()

    # test data
    raw_data = pd.read_csv("detect_ai_content/ml_logic/for_texts/test_data/new_dataset.csv")
    raw_data = raw_data.head(5)

    df = internal.main(raw_data)
    df.to_csv("detect_ai_content/ml_logic/for_texts/test_data/test_output_internalfeature.csv", index=False)
