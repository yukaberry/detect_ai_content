import pandas as pd
import requests
import json
from prefect import task, flow
from sklearn.metrics import accuracy_score


from ml_flow_params import *
from detect_ai_content.ml_logic.for_images.vgg16 import *
from detect_ai_content.ml_logic.for_images.TrueNetImageUsinCustomCNN import *

# TODO
# prefect
# to retrain model
# preproces data

# 0. cloud data souce
# 1. evaluate_model add bigger test dataset
# 2. send_msg get slack API


class PrefectImages():

    @task
    def __init__(self):
        pass

    @task
    def load_data(self):
        # Load dataset

        pass

    @task
    def preprocess_data(self):
        # preprocess data

        pass

    @task
    def load_models(self, model_name):

        model_name

        pass

    @task
    def evaluate_model(self, pred):
        # Evaluate the model on test data

        # TODO
        # update for bigger dataset
        y_test = [0]
        accuracy = accuracy_score(y_test, [pred])
        return accuracy

    @task
    def compare_models(self, results):
        # Compare accuracy scores of different models
        best_model = max(results, key=lambda item: item["accuracy"])
        return best_model


    @task
    def choose_model(self):

        pass

    @task
    def send_msg():

        """
        - send a message to group chat on slack

        # TODO
        # suport error or warning/alart in case of error/code breaks

        """
        # Replace with your actual Slack bot token
        slack_token = 'xoxb-your-slack-token'
        # Replace with your actual channel ID
        channel_id = 'C01234567'
        url = 'https://slack.com/api/chat.postMessage'

        # TODO
        msg = 'Nonthing at this monent and coming soon .... !'

        # Set up headers and payload for Slack API
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {slack_token}'
        }
        data = {
            'channel': channel_id,
            'text': msg
        }

        # Send the message
        # response = requests.post(url,
        #                          headers=headers,
        #                          data=json.dumps(data))

        # response.raise_for_status()

        print(Fore.BLUE + f"\nmessage is sending to our slack channel soon!" + Style.RESET_ALL)
        # return response.json()  # Return Slack's response for confirmation


    @flow(name=PREFECT_FLOW_NAME)
    def main_flow(self, img):

        """
        Prefect workflow

        1. preprocess and load data
        2. load models
        3. train models
        4. compute peformance metrics
        5. evaluate models
        4. compare models
        6. pick better model
        7. send a message to group chat

        """
        # # preprocess data
        # # load data
        # # df = self.load_data()
        # processed_df = self.preprocess_data()

        # # load models-
        # models = self.load_models()

        # # evaluate models
        # evaluate_results = self.evaluate_models()

        # # comapre models
        # model_comparison = self.compare_models()

        # # return best model
        # best_model = self.choose_model()

        # # send a message
        # msg = self.send_msg()

        vgg16 = Vgg16()
        prediction1, message1 = vgg16.predict(img)
        cnn = TrueNetImageUsinCustomCNN()
        prediction2, message2 = cnn.predict(img)

        predictions = [prediction1, prediction2]
        models = [vgg16.name, cnn.name]
        print(models)

        results = []
        for model, pred in zip(models, predictions):
            accuracy = self.evaluate_model(pred)
            results.append({"model": model, "accuracy": accuracy})

        best_model = self.compare_models(results)

        print(f'best model is {best_model}')
        return {'best model': best_model}




if __name__ == '__main__':

    img = Image.open('detect_ai_content/interface/test_img2.jpg')
    #img = Image.open('detect_ai_content/interface/test_img.jpg')

    pi = PrefectImages()
    pi.main_flow(img)
