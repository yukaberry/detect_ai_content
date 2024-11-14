import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords, if specified
        if self.remove_stopwords:
            text = ' '.join(word for word in text.split() if word not in self.stop_words)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def apply_preprocessing(self, df, text_column='text'):
        # Apply preprocessing to the text column
        df['cleaned_text'] = df[text_column].apply(lambda x: self.preprocess_text(x))
        # Select only the necessary columns
        cleaned_df = df[['cleaned_text', 'generated']]
        return cleaned_df
