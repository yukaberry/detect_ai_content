import pandas as pd
import os

from create_internal_features import InternalFeatures

class LightGbm:

    def local_trained_pipeline(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = f'{module_dir_path}/../detect_ai_content/models/leverdewagon/{self.mlflow_model_name}_pipeline.pickle'
        return pickle.load(open(model_path, 'rb'))

    def load_model():

        """
        - Return a model
        - Return None (but do not Raise) if no model is found
        """

        model= None

        return model

    def __init__(self):
        self.description = "XgBoost model 13 nov updated"
        self.name = "XgBoost model"
        self.model = XgBoost.load_model()
        # TODO
        # self.pipeline = self.local_trained_pipeline()



    def get_internal_features(self):

        """
        load df with all of features which are created in create_internal_features.py

        """

        internal_features = InternalFeatures()
        df = internal_features.main()

        return df


    def pre_process(self):
        pass


    def predict(self):

        """
        Return
        1. prediciton class '1' or '0'
        2. prediction message 'AI' or ' Human'

        """
        # get data with freatures
        df = XgBoost.get_internal_features()

        # predict

        # predict class
        # predicted_class = int(predict_proba > 0.5)

        # if predicted_class == 1:
        #     prediction_message = "Predicted as AI"
        # elif predicted_class == 0:
        #     prediction_message = "Predicted as Human"

        # return predicted_class, prediction_message
        pass


# local test
if __name__ == '__main__':

    xgboost = XgBoost()
    # test input data (user input )
    # test_data = pd.read_csv()
    test_text = 'I am from Paris but live in Munich at the moment. I dont like German food. I want to go back to Paris... '

    text_df = pd.DataFrame(data=[test_text],columns=['text'])
    prediction, message = xgboost.predict(text_df)
    print(f"Prediction: {prediction}, Message: {message}")
