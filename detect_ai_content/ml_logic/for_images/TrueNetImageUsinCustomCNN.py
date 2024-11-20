
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow.keras.models as keras_models
import os
from io import BytesIO
from PIL import Image

class TrueNetImageUsinCustomCNN:

    """
    Ab's model

    'cnn_model_4-11_0.1.h5'

    """


    def load_model():
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        model_path = os.path.join(f'{module_dir_path}/../detect_ai_content', 'models','ab', 'cnn_model_4-11_0.1.h5')
        # Should update this as below
        #model_path = os.path.join(f'{module_dir_path}/../detect_ai_content', 'models','aban371818', 'CNN-FINAL-MODEL.h5')

        model = keras_models.load_model(model_path)
        return model

    def __init__(self):
        self.description = "Custom CNN - 7,177 trainable parameters"
        self.name = "TrueNetImageUsinCustomCNN"
        self.model = TrueNetImageUsinCustomCNN.load_model()

    #def pre_process(image_path):
        # img = image.load_img(image_path, target_size=(120, 120))
    def pre_process(img):

        # Resize to 120, 120
        img = img.resize((120, 120))
        img_array = image.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict (self, image_path):
        to_predict_img_array = TrueNetImageUsinCustomCNN.pre_process(image_path)

        # predict probablity
        # output is a probability value (between 0 and 1)
        prediction_results = self.model.predict(to_predict_img_array)
        class_prediction = prediction_results[0][0]
        print(prediction_results)
        predict_prob_confidence = abs(class_prediction - 0.5)/0.5

        # predict class
        predicted_class = int(class_prediction > 0.5)

        # 0 likely representing 'FAKE' and 1 representing 'REAL'
        # TODO clarify
        if predicted_class == 0:
            prediction_message = "Predicted as AI"
        elif predicted_class == 1 :
            prediction_message = "Predicted as Human"

        return predicted_class, prediction_message, predict_prob_confidence


# local test
# if __name__ == '__main__':
#     cnn = TrueNetImageUsinCustomCNN()
#     img = Image.open('test_img2.jpg')
#     prediction, message = cnn.predict(img)
#     print(f"Prediction: {prediction}, Message: {message}")
