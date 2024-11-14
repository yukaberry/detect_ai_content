import pandas as pd
import pickle
import os
from create_internal_features import InternalFeatures

class XgBoost:
    def __init__(self):
        self.description = "XgBoost model 13 nov updated"
        self.name = "XgBoost model"
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'linchenpal', 'best_xgb_model.pkl')
        self.model = self.load_model()

    def load_model(self):
        'Load the pre-trained XGBoost model from pickle file'
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Failed to load model. Error: {e}")
            return None

    def get_internal_features(self, text):
        """Generate a DataFrame of features from the given text."""
        internal_features = InternalFeatures()
        df = internal_features.main(text)
        return df

    def predict(self, text):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        df = self.get_internal_features(text)
        if self.model:
            prediction = self.model.predict(df)[0]
            message = "AI-generated" if prediction == 1 else "Human-generated"
            return prediction, message
        else:
            return None, "Model is not loaded."

# Example of how to use this class
if __name__ == '__main__':
    xgboost = XgBoost()
    test_text = 'I am from Paris but live in Munich at the moment. I dont like German food. I want to go back to Paris... '
    prediction, message = xgboost.predict(test_text)
    print(f"Prediction: {prediction}, Message: {message}")
