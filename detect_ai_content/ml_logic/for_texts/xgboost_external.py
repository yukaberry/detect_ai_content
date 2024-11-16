import pandas as pd
import pickle
import os
from detect_ai_content.ml_logic.for_texts.create_external_features import ExternalFeatures
from TextPreprocessor import TextPreprocessor

class XGBoostExternal:
    def __init__(self):
        self.description = "XGBoost model for external features"
        self.name = "XGBoost External Model"
        # Adjust the model path based on the relative path in your project
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'linchenpal', 'xgboost_external.pkl')
        self.model = self.load_model()
        self.external_features = ExternalFeatures()
        self.text_preprocessor = TextPreprocessor()  # Initialize TextPreprocessor for cleaning text

    def load_model(self):
        """Load the pre-trained XGBoost model from a pickle file."""
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"Failed to load model. Error: Model file not found at {self.model_path}")
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



    def predict(self, text):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        df = self.get_external_features(text)

        # Print the columns of df to debug
        print("Columns in external_df for prediction:", df.columns.tolist())

        if self.model:
            prediction = self.model.predict(df)[0]
            message = "AI-generated" if prediction == 1 else "Human-generated"
            return prediction, message
        else:
            return None, "Model is not loaded."

# Local test
if __name__ == '__main__':
    # Instantiate the XGBoostExternal class
    xgboost_external = XGBoostExternal()

    # Load the test data
    try:
        base_path = os.path.dirname(__file__)
        external_df_path = os.path.join(base_path, "test_data", "external_df.csv")
        external_df = pd.read_csv(external_df_path)
        print(f"Loaded {len(external_df)} rows from 'external_df.csv'")
    except FileNotFoundError:
        print("Error: 'external_df.csv' not found. Run 'create_external_features.py' first!.")
        exit()

    # Drop 'generated' column if it exists
    if 'generated' in external_df.columns:
        external_df = external_df.drop(columns=['generated'])

    # Predict using the loaded data
    if not external_df.empty and xgboost_external.model:
        predictions = xgboost_external.model.predict(external_df)
        print(f"Predictions: {predictions}")
    else:
        print("Model is not loaded or external_df is empty.")
