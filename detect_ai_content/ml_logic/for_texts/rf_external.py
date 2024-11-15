# rf_external.py

import pandas as pd
import pickle
import os

class RandomForestExternal:
    def __init__(self):
        self.description = "Random Forest model for external features"
        self.name = "Random Forest external model"
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'linchenpal', 'rf_tuned_external.pkl')
        self.model = self.load_model()

    def load_model(self):
        """Load the pre-trained Random Forest model from the pickle file."""
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Failed to load model. Error: {e}")
            return None

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
    features_path = os.path.join(os.path.dirname(__file__), '..', 'test_data', 'test_output_externalfeatures.csv')
    features_df = pd.read_csv(features_path)

    # Initialize the model
    rf_external = RandomForestExternal()

    # Run prediction on the first row of the features
    prediction, message = rf_external.predict(features_df.iloc[[0]])

    # Display results
    print(f"Prediction: {prediction}, Message: {message}")
