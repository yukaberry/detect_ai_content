
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras.models as keras_models
import os

class TrueNetImageUsinCustomCNN:
    def load_model():
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = os.path.join(f'{module_dir_path}/../detect_ai_content', 'models', 'cnn_model_4-11_0.1.h5')
        model = keras_models.load_model(model_path)
        return model

    def __init__(self):
        self.description = "Custom CNN - 7,177 trainable parameters"
        self.name = "TrueNetImageUsinCustomCNN"
        self.model = TrueNetImageUsinCustomCNN.load_model()

    def pre_process(image_path):
        img = image.load_img(image_path, target_size=(120, 120))
        img_array = image.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict (self, image_path):
        to_predict_img_array = TrueNetImageUsinCustomCNN.pre_process(image_path)
        prediction = self.model.predict(to_predict_img_array)
        if prediction[0][0] < 0.5:
            return " This is Fake :( "
        return " Real it is :D "
