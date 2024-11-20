# current version: 0.2 on 19.11.2024

import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras.models as keras_models
import detect_ai_content
import os
import math

class image_classifier_cnn:

    def __init__(self):
        self.description = "Trained CNN model to classify images as Real or AI generated"
        self.name = "image_classifier_cnn"
        self.model = self.load_model()

    def load_model(self):
        module_dir_path = os.path.dirname(os.path.abspath(detect_ai_content.__file__))
        model_path = os.path.join(module_dir_path, 'models', 'aban371818' ,'CNN-FINAL-MODEL.h5')
        model = keras_models.load_model(model_path)
        return model

    def pre_process(self,image_path):

        img = image.load_img(image_path, target_size=(225, 225))
        #img = img.resize((225, 225))
        img_array = image.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):

        """
        - preprocess input img data
        - retun 3 values
        1. prediction class (1 or 0)
        2. message (Ai or Human)
        3. Probability (2 dicimal)

        """
        to_predict_img_array = self.pre_process(image_path)

        prediction_results = self.model.predict(to_predict_img_array)
        class_prediction = float(prediction_results[0][0])
        print(prediction_results)

        predict_prob_confidence = round(abs(class_prediction - 0.5)/0.5, 2)

        # # predict class
        predicted_class = int(class_prediction > 0.5)

        # 0 likely representing 'FAKE' and 1 representing 'REAL'
        if predicted_class == 0:
            prediction_message = "Predicted as AI"
        elif predicted_class == 1 :
            prediction_message = "Predicted as Human"

        return predicted_class, prediction_message, predict_prob_confidence


        # if round(float(prediction[0][0]) * 100, 2) > 0.5:
            #return "The image is probably real :D"
        # return "Ah ha! You've uploaded an AI image!"
        #return f"Likelihood of your image being real is: {round(float(prediction[0][0]) * 100, 2)} "
