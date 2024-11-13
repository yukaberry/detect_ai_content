import pandas as pd
import numpy as np
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

from slang_dict import *

class InternalFeature():

    # def load_model(self):
    #     pass
    # def __init__(self ):
    #     pass

    def text_based_features(self, text_df, text_column='text'):

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

        # TODO
        # change data type

        # TODO
        # COMPUTATION SUPER HEAVY!!!

        features = pd.DataFrame()

        # Initialize sentiment analysis pipeline with GPU (device=0)
        # TODO
        # local test run (device=1)
        sentiment_pipeline = pipeline('sentiment-analysis', device=-1, truncation=True, max_length=512)

        # Apply sentiment analysis in batches
        texts = text_df[text_column].tolist()
        sentiment_results = sentiment_pipeline(texts, truncation=True, max_length=512, batch_size=32)

        # Extract sentiment labels from the results
        features['sentiment'] = [result['label'] for result in sentiment_results]

        # Polarity and subjectivity from TextBlob
        features['polarity'] = text_df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
        features['subjectivity'] = text_df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)

        return features


    def ngrams_keyword_features(self, text_df, text_column='text'):

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

        # TODO
        # datatype change
        features = pd.DataFrame()

        # Process text in batches to improve efficiency
        docs = nlp.pipe(text_df[text_column], batch_size=32)  # Adjust batch size as needed for performance
        features['dependency_count'] = [len(doc.ents) for doc in docs]

        return features


    def spelling_error_features(self, text_df, text_column='text'):

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

        # TODO
        # change datatype
        features = pd.DataFrame()

        # Line break count
        features['line_break_count'] = text_df[text_column].apply(lambda x: x.count('\n'))

        # Punctuation count
        features['punctuation_count'] = text_df[text_column].apply(lambda x: sum([1 for char in x if char in '.,;!?']))

        return features

    def count_slang(self, text, slang_dict):

        # maybe you need to add cleaning test function here below
        # it will depend on txt dataset
        # ex) text = clean_txt(text)

        slang_count = sum(1 for word in text if word in slang_dict)
        return slang_count


    def concat_features(self, data, raw_data):

        internal_features = pd.concat([
                            # text_features,
                            # lexical_features,
                            # pos_features,
                            # sentiment_features,
                            # ngram_features,
                            # complexity_features,
                            # structural_features,
                            # repetition_features,
                            # spelling_error_features
                            # slang
                            data
                            ], axis=1)

        internal_df = pd.concat([raw_data[['generated', 'text']],
                                 internal_features],
                                axis=1)

        #test
        print(internal_df.shape)

        return internal_df


    # def preprocess():

    #     pass

    # def predict():

    #     pass

    # def main():
    #     pass
        # self.loadmodel()
        # .....


        # return pred, msg


# local test
if __name__ == '__main__':
    internal = InternalFeature()

    # test data
    raw_data = pd.read_csv("test_data/new_dataset.csv")
    # feature = internal.text_based_features(raw_data)
    # print(feature)
    # feature2 = internal.lexical_diversity_readability(raw_data)
    # print(feature2)

    # feature3 = internal.pos_tagging_features(raw_data)
    # print(feature3)

    # TODO
    # COMPUTATION SUPER HEAVY!!!

    # feature4 = internal.sentiment_emotion_features(raw_data)
    # print(feature4)

    # feature5 = internal.ngrams_keyword_features(raw_data)
    # print(feature5)

    feature6 = internal.linguistic_complexity_features(raw_data)
    print(feature6)

    # feature7 = internal.spelling_error_features(raw_data)
    # print(feature7)

    # feature8 = internal.repetition_features(raw_data)
    # print(feature8)

    # feature9 = internal.structural_formatting_features(raw_data)
    # print(feature9)

    # feature10 = internal.count_slang(raw_data)
    # print(feature10)


    # prediction, message = internal.predict(data)
    # print(f"Prediction: {prediction}, Message: {message}")
