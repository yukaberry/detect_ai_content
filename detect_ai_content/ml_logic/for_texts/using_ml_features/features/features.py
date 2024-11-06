
from textblob import TextBlob

from nltk.corpus import stopwords

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
from nltk import tokenize

english_stopwords = stopwords.words('english')

import string
import re

def extract_sentences(text):
    return tokenize.sent_tokenize(text)

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
# 710   ⭐️  https://github.com/barrust/pyspellchecker
# 68    ⭐️  https://github.com/hellohaptik/spello
# 9000  ⭐️  https://github.com/sloria/TextBlob
# 500   ⭐️  https://github.com/filyp/autocorrect
# 612       Jamspell

from nltk.corpus import wordnet
from nltk.corpus import words
wordnet_words = list(wordnet.words())
words_dict = {}
for w in words.words():
    words_dict[w]=w

for w in list(wordnet_words):
    words_dict[w]=w

from nltk.stem import *
stemmer = PorterStemmer()
from textblob import Word

def compute_number_of_text_corrections_using_nltk_words(text):
    text_blob = TextBlob(text)
    corrections = 0
    ignore_words = [
        'n\'t'
    ]
    for sentence in text_blob.sentences:
        for word in sentence.words:
            if len(word) > 2 and word not in ignore_words:
                word_lower = str.lower(word)
                if stemmer.stem(word_lower) not in words_dict:
                    singular_form = Word(word_lower).singularize()
                    if singular_form not in words_dict:
                        # print(f'❌ {word_lower} - sing({singular_form})')
                        corrections+= 1
#         else:
#            print(f'✅ {word_lower}')

    return corrections

def number_of_sentences(text):
    text_blob = TextBlob(text)
    return len(text_blob.sentences)

def text_lenght(text):
    return len(text)
