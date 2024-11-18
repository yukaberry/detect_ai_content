import os
import pandas as pd
import pickle
from detect_ai_content.ml_logic.for_texts.create_internal_features import InternalFeatures
import lightgbm #important to keep, to be able to load the pickle file
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler

class LgbmInternal:
    def __init__(self):
        self.description = "LightGBM Internal Model for AI Content Detection"
        self.name = "LightGBM Internal Model"

        # Adjust paths for model and features
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        self.model_path = f'{module_dir_path}/../detect_ai_content/models/linchenpal/lgbm_internal.pkl'
        self.model = self.load_model()

    def st_size(self):
        import os
        return os.stat(self.model_path).st_size

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
        return df
        # return df.drop(columns=['generated'], errors='ignore')  # Drop 'generated' if present

    def predict(self, text):
        """Predict and return the class ('1' for AI or '0' for Human) and the corresponding message."""
        df = self.get_internal_features(text)
        if self.model:
            prediction = self.model.predict(df)[0]
            message = "AI-generated" if prediction == 1 else "Human-generated"
            return prediction, message
        else:
            return None, "Model is not loaded."

    def pretrained_model(self):
        pipeline = Pipeline([
            ('enricher', LgbmGenerateInternalFeaturesTransformer()),
            ('estimator', self.model),
        ])
        return pipeline

def LgbmGenerateInternalFeaturesFunction(df):
    internal_features = InternalFeatures()
    enriched_df = internal_features.process(df)
    return enriched_df

def LgbmGenerateInternalFeaturesTransformer():
    return FunctionTransformer(LgbmGenerateInternalFeaturesFunction)


# Local test
if __name__ == '__main__':
    lgbm_int = LgbmInternal()
    test_text = 'I am from Bogota but live in Munich at the moment. I don’t like German food. I want to go back to Montevideo...'
    prediction, message = lgbm_int.predict(test_text)
    print(f"Prediction: {prediction}, Message: {message}")
