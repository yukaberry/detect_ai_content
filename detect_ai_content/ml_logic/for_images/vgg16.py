import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
import tensorflow as tf

from io import BytesIO
from PIL import Image
import os


from colorama import Fore, Style

class Vgg16:

    def load_model():

        """
        - Return a keras VGG16 model
        - Return None (but do not Raise) if no model is found
        """

        model= None

        model = VGG16(weights="imagenet",
                       include_top=False,
                       input_shape=(224, 224, 3))
        model.trainable = False  # Freeze the VGG16 layersth)
        x = layers.Flatten()(model.output)
        x = layers.Dense(256, activation='relu')(x)
        num_classes = 1
        x = layers.Dense(num_classes, activation='sigmoid')(x)

        x = layers.Dropout(0.5)(x)
        model = Model(inputs=model.input, outputs=x)
        model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
        return model

    def __init__(self):
        self.description = "prerained keras model"
        self.name = "vgg16"
        self.model = Vgg16.load_model()

    def pre_process(img):


        # Resize to 224 x 224 for vgg16 model
        img = img.resize((224, 224))
        # Convert the image pixels to a numpy array
        arr = img_to_array(img)
        # reshape
        arr = arr.reshape((1, 224, 224, 3))
        # Prepare the image for the VGG model
        arr = preprocess_input(arr)

        return arr

    def predict(self, user_input):

        """
        Return
        1. prediciton class '1' or '0'
        2. prediction message 'AI' or ' Human'

        """
        # preprocess img
        to_predict_img_array = Vgg16.pre_process(user_input)

        # predict probablity
        # output is a probability value (between 0 and 1)
        predict_proba = self.model.predict(to_predict_img_array)[0][0]

        # Get predicted indices
        # do not use np.argmax for binary, it is for multi-class!!
        # predicted_probabilities = np.argmax(predicted_class, axis=1)

        # predict class
        predicted_class = int(predict_proba > 0.5)

        if predicted_class == 1:
            prediction_message = "Predicted as AI"
        elif predicted_class == 0:
            prediction_message = "Predicted as Human"

        return predicted_class, prediction_message


# local test
# if __name__ == '__main__':
#     vgg16 = Vgg16()
#     img = Image.open('test_img2.jpg')
#     prediction, message = vgg16.predict(img)
#     print(f"Prediction: {prediction}, Message: {message}")
