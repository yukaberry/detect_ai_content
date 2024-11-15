import re
import string
from nltk.corpus import stopwords
import nltk
import pandas as pd

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

        # Select only the 'cleaned_text' column and 'generated' if it exists
        if 'generated' in df.columns:
            cleaned_df = df[['cleaned_text', 'generated']]
        else:
            cleaned_df = df[['cleaned_text']]

        return cleaned_df


    # Local test
if __name__ == '__main__':
    text_preprocessor = TextPreprocessor()
    sample_data = pd.DataFrame({"text": ["This is a sample sentence for preprocessing.", "Another example sentence here."]})
    processed_data = text_preprocessor.apply_preprocessing(sample_data)
    print(processed_data)
