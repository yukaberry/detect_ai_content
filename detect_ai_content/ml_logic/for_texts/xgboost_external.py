import pandas as pd
import pickle
import os
from create_external_features import ExternalFeatures

class XGBoostExternal:
    def __init__(self):
        self.description = "XGBoost model for external features"
        self.name = "XGBoost External Model"
        # Adjust the model path based on the relative path in your project
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'linchenpal', 'xgboost_external.pkl')
        self.model = self.load_model()
        self.external_features = ExternalFeatures()

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

        # Process the DataFrame to get external features
        df_features = self.external_features.process(df)

        # Drop 'generated' column if it exists in the DataFrame for prediction
        if 'generated' in df_features.columns:
            df_features = df_features.drop(columns='generated')

        return df_features

    def predict(self, text):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        df = self.get_external_features(text)

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

    # Sample test input
    test_text = "This is an example sentence to test if it is AI-generated or human-generated."

    # Make prediction
    prediction, message = xgboost_external.predict(test_text)
    print(f"Prediction: {prediction}, Message: {message}")
