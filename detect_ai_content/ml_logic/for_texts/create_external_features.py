import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class ExternalFeatures:
    def __init__(self):
        # Initialize SBERT model
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize clustering model
        self.kmeans = None

    def generate_sbert_embeddings_with_pca(self, df, text_column='cleaned_text', n_components=50):
        # Convert text to embeddings
        embeddings = df[text_column].apply(lambda x: self.sbert_model.encode(x)).tolist()
        embedding_df = pd.DataFrame(embeddings)

        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embedding_df)
        reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'pca_embedding_{i}' for i in range(n_components)])

        return pd.concat([df.reset_index(drop=True), reduced_df], axis=1)

    def calculate_cosine_similarity(self, df, embeddings_column_prefix='pca_embedding_', n_components=50):
        # Extract embeddings columns
        embedding_columns = [f'{embeddings_column_prefix}{i}' for i in range(n_components)]
        embeddings = df[embedding_columns].astype(np.float32).values

        # Apply K-Means clustering
        self.kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=100).fit(embeddings)
        cluster_labels = self.kmeans.labels_
        cluster_centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)

        # Calculate cosine similarity
        similarities = []
        for i in range(len(embeddings)):
            text_embedding = torch.tensor(embeddings[i], dtype=torch.float32)
            cluster_center = cluster_centers[cluster_labels[i]]
            similarity = torch.nn.functional.cosine_similarity(text_embedding.unsqueeze(0), cluster_center.unsqueeze(0)).item()
            similarities.append(similarity)

        df['cosine_similarity'] = similarities
        df['cluster_label'] = cluster_labels
        return df

    def generate_topics_lda(self, df, text_column='cleaned_text', n_topics=5):
        # Vectorize text
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(df[text_column])

        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_matrix = lda.fit_transform(doc_term_matrix)
        lda_df = pd.DataFrame(lda_matrix, columns=[f'lda_topic_{i}' for i in range(n_topics)])

        return pd.concat([df.reset_index(drop=True), lda_df], axis=1)

    def process(self, df):
        # Apply SBERT with PCA embeddings
        df = self.generate_sbert_embeddings_with_pca(df)

        # Calculate cosine similarity based on PCA embeddings
        df = self.calculate_cosine_similarity(df)

        # Generate LDA topics
        df = self.generate_topics_lda(df)

        # Rename the resulting DataFrame as external_df
        external_df = df

        return external_df

# local test

import pandas as pd
from create_external_features import ExternalFeatures
from TextPreprocessor import TextPreprocessor

if __name__ == '__main__':
    # Initialize the TextPreprocessor and ExternalFeatures classes
    text_preprocessor = TextPreprocessor()
    external = ExternalFeatures()

    # Load the test data
    raw_data = pd.read_csv("detect_ai_content/ml_logic/for_texts/test_data/new_dataset.csv")

    # Apply preprocessing to the text
    processed_data = text_preprocessor.preprocess_dataframe(raw_data)

    # Extract external features
    df = external.process(processed_data)

    # Save the result to a CSV file for verification
    df.to_csv("detect_ai_content/ml_logic/for_texts/test_data/test_output_externalfeatures.csv", index=False)

    print("External features have been successfully saved to 'test_output_externalfeatures.csv'")
