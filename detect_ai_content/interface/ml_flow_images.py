import pandas as pd
import requests
import json
from prefect import task, flow

from ml_flow_params import *
from ml_logic.for_images.vgg16 import *

class PrefectImages():

    @task
    def __init__(self):
        pass

    @task
    def load_data(self):
        pass

    @task
    def preprocess_data(self):

        pass

    @task
    def load_models(self):

        pass

    @task
    def evaluate_models(self):

        pass

    @task
    def compare_models(self):
        pass

    @task
    def choose_model(self):

        pass

    @task
    def send_msg():

        """
        send a message to group chat on slack

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
    def main_flow(self):

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


        # preprocess data
        # load data
        df = self.load_data()
        processed_df = self.preprocess_data()

        # load models-
        models = self.load_models()

        # evaluate models
        evaluate_results = self.evaluate_models()

        # comapre models
        model_comparison = self.compare_models()

        # return best model
        best_model = self.choose_model()

        # send a message
        msg = self.send_msg()

        return


if __name__ == '__main__':
    pi = PrefectImages()
    pi.main_flow()
