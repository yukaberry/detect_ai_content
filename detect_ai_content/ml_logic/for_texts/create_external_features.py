import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import torch
import numpy as np
from detect_ai_content.ml_logic.for_texts.TextPreprocessor import TextPreprocessor

class ExternalFeatures:
    def __init__(self):
        # Initialize SBERT model and clustering model
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kmeans = None

    def generate_sbert_embeddings_with_pca(self, df, text_column='cleaned_text', n_components=50):
        # Generate SBERT embeddings and apply PCA
        embeddings = df[text_column].apply(lambda x: self.sbert_model.encode(x)).tolist()
        embedding_df = pd.DataFrame(embeddings)

        # Adjust n_components to be the minimum of 50 or the number of features in embeddings
        n_components = min(n_components, embedding_df.shape[1])
        pca = PCA(n_components=n_components)

        reduced_embeddings = pca.fit_transform(embedding_df)
        reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'pca_embedding_{i}' for i in range(n_components)])

        # Concatenate the original dataframe with PCA embeddings
        return pd.concat([df.reset_index(drop=True), reduced_df], axis=1)

    def calculate_cosine_similarity(self, df, embeddings_column_prefix='pca_embedding_', n_components=50):
        # Calculate cosine similarity of PCA embeddings with K-Means clustering
        embedding_columns = [f'{embeddings_column_prefix}{i}' for i in range(n_components)]
        embeddings = df[embedding_columns].astype(np.float32).values
        self.kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=100).fit(embeddings)
        cluster_labels = self.kmeans.labels_
        cluster_centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
        similarities = [torch.nn.functional.cosine_similarity(
            torch.tensor(embeddings[i], dtype=torch.float32).unsqueeze(0),
            cluster_centers[cluster_labels[i]].unsqueeze(0)).item()
            for i in range(len(embeddings))]
        df['cosine_similarity'] = similarities
        df['cluster_label'] = cluster_labels
        return df

    def generate_topics_lda(self, df, text_column='cleaned_text', n_topics=5):
        # Generate topics using LDA
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(df[text_column])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_matrix = lda.fit_transform(doc_term_matrix)
        lda_df = pd.DataFrame(lda_matrix, columns=[f'topic_{i}' for i in range(n_topics)])
        return pd.concat([df.reset_index(drop=True), lda_df], axis=1)

    def process(self, df):
        # Apply SBERT with PCA embeddings
        df = self.generate_sbert_embeddings_with_pca(df)

        # Calculate cosine similarity
        df = self.calculate_cosine_similarity(df)

        # Generate LDA topics
        df = self.generate_topics_lda(df)

        # Define the correct column order for external_df
        feature_columns = [f'pca_embedding_{i}' for i in range(50)] + \
                        ['cosine_similarity', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4']

        # Filter relevant columns and reorder them according to the specified order
        return df[feature_columns]


    # local test
import os
if __name__ == '__main__':
    # Initialize the TextPreprocessor and ExternalFeatures classes
    text_preprocessor = TextPreprocessor()
    external = ExternalFeatures()

    # Define relative path to the test data
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "test_data", "new_dataset.csv")
    output_path = os.path.join(base_path, "test_data", "external_df.csv")

    # Load the test data
    try:
        raw_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file 'new_dataset.csv' was not found in {data_path}.")
        exit()

    # Apply preprocessing to the text
    processed_data = text_preprocessor.apply_preprocessing(raw_data)

    # Extract external features
    try:
        df = external.process(processed_data)
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        exit()

    # Save the result to a CSV file for verification
    try:
        df.to_csv(output_path, index=False)
        print(f"External features have been successfully saved to '{output_path}'")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")
