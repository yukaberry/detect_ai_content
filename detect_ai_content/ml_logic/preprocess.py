import pandas as pd

def text_lenght(text):
    return len(text)

def average_sentences(text):
    return len(text.split("."))

def preprocess_text(text):
    X = pd.DataFrame(dict(
        text=[text],
    ))

    X_processed = preprocess_features(X)
    print(X_processed)
    return X_processed

def preprocess_features(df):
    texts_df = df.copy()
    texts_df['text_lenght'] = texts_df['text'].apply(text_lenght)
    texts_df['sentences_count'] = texts_df['text'].apply(average_sentences)
    texts_df = texts_df.drop(columns=['text'])
    return texts_df
