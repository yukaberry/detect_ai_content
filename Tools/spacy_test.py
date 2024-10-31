
import sys
import os

import pandas as pd

# from detect_ai_content.ml_logic.for_texts.using_ml_features.using_ml_features import enrich

import spacy
nlp = spacy.load('en_core_web_sm')

import contextualSpellCheck
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)
nlp.add_pipe('sentencizer')

texts = [
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
    "lksjdfl lsdkfj lskdjf lksdjf lksjdflksdjf lksdjf ",
]

for doc in nlp.pipe(texts, batch_size=8, disable=['parser', 'tagger', 'ner']):
    print(len(doc._.suggestions_spellCheck))
