
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


import torch
from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

def compute_masked_words_BERT_prediction(text):
    text_blob = TextBlob(text)
    number_of_test = 0
    number_of_correct_prediction = 0

    for sentence in text_blob.sentences:
        if len(sentence) > 500:
            # print('ignore the sentence')
            continue

        for word in sentence.words:
            if len(word) > 5:
                masked_sentence = sentence.replace(word, "<mask>")
                # print(f'START_{masked_sentence}_END')
                top_k = 10
                top_clean = 5
                input_ids, mask_idx = BERT_encode(bert_tokenizer, f'{masked_sentence}')
                with torch.no_grad():
                    predict = bert_model(input_ids)[0]
                predict_words = BERT_decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
                # print(predict_words)

                number_of_test += 1
                if word in predict_words:
                    number_of_correct_prediction +=1

    # print(f"number_of_test: {number_of_test}")
    # print(f"number_of_correct_prediction: {number_of_correct_prediction}")
    # print(f"prediction : {round(100 * number_of_correct_prediction/number_of_test)}%")
    return (number_of_test, number_of_correct_prediction)

def BERT_decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def BERT_encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx
