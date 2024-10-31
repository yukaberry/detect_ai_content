
from textblob import TextBlob

from nltk.corpus import stopwords

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('words')
from nltk import tokenize

import spacy
import contextualSpellCheck
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)

english_stopwords = stopwords.words('english')

import string
import re

def extract_sentences(text):
    return tokenize.sent_tokenize(text)

def spacy_sentences(text):
    return nlp(text).sents

def compute_punctuation_in_text(text):
    number_of_ponctuations = 0

    for c in text:
        if c in string.punctuation:
            number_of_ponctuations += 1

    return number_of_ponctuations

def _compute_average_sentiment_polarity_in_text(text, polarity="neg"):
    blob = TextBlob(text)

    global_polarity = 0
    for sentence in blob.sentences:
        if polarity == "neg" and sentence.sentiment.polarity < 0:
            global_polarity += sentence.sentiment.polarity
        if polarity == "pos" and sentence.sentiment.polarity > 0:
            global_polarity += sentence.sentiment.polarity
    return global_polarity

def compute_neg_sentiment_polarity_in_text(text):
    return _compute_average_sentiment_polarity_in_text(text, polarity="neg")

def compute_pos_sentiment_polarity_in_text(text):
    return _compute_average_sentiment_polarity_in_text(text, polarity="pos")

def compute_repetitions_in_text(text):
    word_occurences = {}
    blob = TextBlob(text)

    for w in blob.words:
        if w not in english_stopwords and w not in string.punctuation:
            if w not in word_occurences:
                word_occurences[w]=0
            word_occurences[w] += 1

    repetions = 0
    for k in word_occurences:
        if word_occurences[k] > 1:
            repetions = repetions + word_occurences[k]

    return repetions

# https://github.com/diffitask/spell-checkers-comparison

def _number_of_corrections_using_Spacy(text):
    # print(f'_number_of_corrections_using_Spacy: {text}')
    doc = nlp(f"{text}.")
    return len(doc._.suggestions_spellCheck)

def compute_number_of_text_corrections(text):
    text_blob = TextBlob(text)
    corrections = 0
    for sentence in text_blob.sentences:
        if len(sentence) < 500:
            corrections += _number_of_corrections_using_Spacy(sentence)

    return corrections

def compute_number_of_corrections_in_sentence(text):
    return _number_of_corrections_using_Spacy(text)

def number_of_sentences(text):
    text_blob = TextBlob(text)
    return len(text_blob.sentences)

def text_lenght(text):
    return len(text)
