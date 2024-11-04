import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

"""
    - Make sure to follow the given directory structure for the below code to work
    - models > model_name.h5

"""

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(ROOT_PATH, 'models', 'cnn_model_4-11_0.1.h5')
model = load_model(model_path)

def pre_process(image_path):

    img = image.load_img(image_path, target_size=(120, 120))
    img_array = image.img_to_array(img)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict (image_path):

    to_predict_img_array = pre_process(image_path)
    prediction = model.predict(to_predict_img_array)
    if prediction[0][0] < 0.5:
        return " This is Fake :( "
    return " Real it is :D "
