# rf_external.py

import pandas as pd
import pickle
import os
from detect_ai_content.ml_logic.for_texts.create_external_features import ExternalFeatures
from detect_ai_content.ml_logic.for_texts.TextPreprocessor import TextPreprocessor


class RandomForestExternal:
    def __init__(self):
        self.description = "Random Forest model for external features"
        self.name = "Random Forest external model"
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'linchenpal', 'rf_external.pkl')
        self.model = self.load_model()
        self.external_features = ExternalFeatures()
        self.text_preprocessor = TextPreprocessor()

    def load_model(self):
        """Load the pre-trained Random Forest model from the pickle file."""
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Failed to load model. Error: {e}")
            return None


    def get_external_features(self, text):
        """Generate a DataFrame of external features from the given text."""
        # Create a DataFrame for the input text
        df = pd.DataFrame({"text": [text]})

        # Clean the text using TextPreprocessor
        cleaned_df = self.text_preprocessor.apply_preprocessing(df)

        # Process the cleaned DataFrame to get external features
        df_features = self.external_features.process(cleaned_df)

        return df_features

    def predict(self, features_df):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        if self.model:
            prediction = self.model.predict(features_df)[0]
            message = "AI-generated" if prediction == 1 else "Human-generated"
            return prediction, message
        else:
            return None, "Model is not loaded."

# Local test
if __name__ == '__main__':


    # Load the external features from the CSV file
    features_path = os.path.join(os.path.dirname(__file__), '..', 'test_data', 'external_df.csv')
    features_df = pd.read_csv(features_path)

    # Initialize the model
    rf_external = RandomForestExternal()

    # Run prediction on the first row of the features
    prediction, message = rf_external.predict(features_df.iloc[[0]])

    # Display results
    print(f"Prediction: {prediction}, Message: {message}")
