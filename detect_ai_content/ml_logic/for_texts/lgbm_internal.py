import os
import pandas as pd
import pickle
from create_internal_features import InternalFeatures

class LgbmInternal:
    def __init__(self):
        self.description = "LightGBM Internal Model for AI Content Detection"
        self.name = "LightGBM Internal Model"
        # Adjust paths for model and features
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lgbm_internal.pkl')
        self.model = self.load_model()

    def load_model(self):
        """Load the LightGBM model from a pickle file."""
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Failed to load model. Error: {e}")
            return None

    def get_internal_features(self, text):
        """Generate a DataFrame of internal features from the given text."""
        internal_features = InternalFeatures()
        # Assume the InternalFeatures class has a method to process single text input
        df = internal_features.process(pd.DataFrame({"text": [text]}))
        return df.drop(columns=['generated'], errors='ignore')  # Drop 'generated' if present

    def predict(self, text):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        df = self.get_internal_features(text)
        if self.model:
            prediction = self.model.predict(df)[0]
            message = "AI-generated" if prediction == 1 else "Human-generated"
            return prediction, message
        else:
            return None, "Model is not loaded."

# Local test
if __name__ == '__main__':
    lgbm_int = LgbmInternal()
    test_text = 'I am from Bogota but live in Munich at the moment. I don’t like German food. I want to go back to Montevideo...'
    prediction, message = lgbm_int.predict(test_text)
    print(f"Prediction: {prediction}, Message: {message}")