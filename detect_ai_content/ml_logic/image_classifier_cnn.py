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
        img_array = image.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        to_predict_img_array = self.pre_process(image_path)
        prediction = self.model.predict(to_predict_img_array)
        return float(prediction[0][0])
