import pandas as pd
import os

import pickle
from detect_ai_content.ml_logic.for_texts.create_internal_features import InternalFeatures
import pdb

class XGB:

    def __init__(self):
        self.description = "XgBoost model 14 nov updated"
        self.name = "XgBoost model (lina)"


    def load_model(self):

        """
        Load the pre-trained XGBoost model from pickle file

        """
        # Specify the path to your model file
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path =f'{module_dir_path}/../detect_ai_content/models/linchenpal/best_xgb_model.pkl'


        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        return model


    def get_internal_features(self, text):

        """Generate a DataFrame of features from the given text."""

        df = InternalFeatures().main(text)

        return df


    def main(self, df):

        """
        - load model
        - predict
        - return prediction and the class ('1' for AI or '0' for Human) and the corresponding message.

        """
        model = None

        #pdb.set_trace()
        model = self.load_model()

        # RETURN ONLY CLASS!
        if model:
            # model.predict(df) : array, len 1
            # prediction datatype : numpy.int64
            predictions = model.predict(df)
            # if prediction == 1:
            #     message = "AI-generated"
            # else:
            #     message = "Human-generated"

            # return prediction, message

        # else:
        #     return None, "Model is not loaded."

        return predictions


# local test
if __name__ == '__main__':

    xgb = XGB()

    # test input data (user input )
    test_text = 'I m from tokyo but living in Berlin'
    raw_df = pd.DataFrame(data=[test_text],columns=['text'])

    df = xgb.get_internal_features(raw_df)

    # predict
    prediction, message = xgb.main(df)
    print(prediction, message)

    ###### test dataframe type #####
    # test dataframe type
    # raw_data = pd.read_csv("detect_ai_content/ml_logic/for_texts/test_data/new_dataset.csv")
    # raw_df = raw_data.head(5)

    # X = df.iloc[:, 1:]
    # y = df['generated']
    # predict
    # pred = xgb.predict(X)

    #pdb.set_trace()
